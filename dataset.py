import torch
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import random

"""
Defines a Pascal VOC dataset for use in YOLOV1 object detector.

__getitem__ returns:
    image: PIL image object
    label_grid: (SxSxC+5 tensor) last dim has form 
        [one hot class_labels...,x_mid,y_mid,width,height,contains_object] 
        where coords are relative to the grid cell in which they are contained
"""

class YOLOVOCDataset(torch.utils.data.Dataset):
    
    def __init__(self, csv_file, img_dir, label_dir, transform, hflip_prob=0,
            random_crops=0,S=7, B=2, C=20,):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.hflip_prob = hflip_prob
        self.random_crops = random_crops

    def __len__(self):
        return len(self.annotations)

    def transform(self, img, label):
        img = self.transform

    def __getitem__(self, index):
        example = self.annotations.iloc[index]

        label_path = os.path.join(self.label_dir, example[1])
        boxes = []
        with open(label_path) as labels:
            for label in labels.readlines():
                class_label, x, y, width, height = [
                    float(x) 
                    for x in label.replace("\n","").split()
                ]
                #we want class_labels as integers, not floats
                boxes.append([int(class_label),x,y,width,height])

        boxes = torch.tensor(boxes)

        img_path = os.path.join(self.img_dir, example[0]) 
        image = Image.open(img_path)

        if self.transform:
            image, _ = self.transform(image, boxes)

        if self.hflip_prob:
            if random.random() > self.hflip_prob: 
                image, boxes = hflip(image, boxes)
        
        if self.random_crops:
            crop_factor = self.random_crops*random.random()
            super_size = int((1+crop_factor)*image.shape[1])
            resize = transforms.Resize((super_size,super_size))
            super_image = resize(image)

            image, boxes = randomCrop(super_image, boxes, (448,448))

        label_grid = torch.zeros(self.S, self.S, self.C + 5)
        for box in boxes:
            """
            convert labels from full image form to YOLO grid form.
                
            IN: [class_label, x_mid, y_mid, width, height] where coords are between 0 and 1
                (relative to the image dimensions)

            OUT: SxSx[one hot class_labels..., x_mid, y_mid, width, height] tensor 
                where coords are relative to the grid space (i,j between 0 and S) 

            grid cells are "responsible" for predicting a bounding box if the center
            of the ground truth box falls within them.
            """
            class_label, x, y, width, height = box.tolist()
            i,j = int(self.S*x), int(self.S*y)
            cell_x, cell_y = self.S*x-i, self.S*y-j
            cell_width, cell_height = (width*self.S, height*self.S)

            # set object present indicator to one in cell responsible for object
            # note that only one object per cell will be included
            if label_grid[i, j, -1] == 0:
                label_grid[i, j, -1] = 1
                box_coordinates = torch.tensor(
                    [cell_x, cell_y, cell_width, cell_height]
                )
                #define box coords, set index of class label to 1
                label_grid[i, j, -5:-1] = box_coordinates
                label_grid[i, j, int(class_label)] = 1
        
        return image, label_grid

def hflip(img, labels):
    img = torch.flip(img, (2,))
    labels[:,1] = 1-labels[:,1]
    return img, labels

def randomCrop(img, labels, output_size=(448,448)):
    """
    inputs:
        img (tensor): [in_channels, x_dim, y_dim]
        labels (tensor): [num_bboxes, (class_label, x_mid, y_mid,
            width, height)])
        output_size (tuple): (x_output_size,y_output_size)
    returns:
        cropped img, labels as tensors
    """
    cropped_corner = random.randrange(4)
    x_output_size, y_output_size = output_size
    x_crop_size = img.shape[1]-x_output_size
    y_crop_size = img.shape[2]-y_output_size
    x_crop_factor = (float(x_crop_size)/img.shape[1])
    y_crop_factor = (float(y_crop_size)/img.shape[2])
    if (x_crop_size < 1 or y_crop_size < 1):
        return img, labels

    if cropped_corner==0:
        img = img[...,:-x_crop_size,:-y_crop_size]
        labels[...,1] = labels[...,1]/(1-x_crop_factor)
        labels[...,2] = labels[...,2]/(1-y_crop_factor)
    elif cropped_corner==1:
        img = img[...,:-x_crop_size,y_crop_size:]
        labels[...,1] = (labels[...,1]-x_crop_factor)/(1-x_crop_factor)
        labels[...,2] = labels[...,2]/(1-y_crop_factor)
    elif cropped_corner==2:
        img = img[...,x_crop_size:,:-y_crop_size]
        labels[...,1] = labels[...,1]/(1-x_crop_factor)
        labels[...,2] = (labels[...,2]-y_crop_factor)/(1-y_crop_factor)
    else:
        img = img[...,x_crop_size:,y_crop_size:]
        labels[...,1] = (labels[...,1]-x_crop_factor)/(1-x_crop_factor)
        labels[...,2] = (labels[...,2]-y_crop_factor)/(1-y_crop_factor)

    labels[...,3] = labels[...,3]/(1-x_crop_factor)
    labels[...,4] = labels[...,4]/(1-y_crop_factor)

    labels = [label for label in labels.tolist()
                if 1.0 > label[1] > 0.0
                and 1.0 > label[2] > 0.0
    ]
    
    labels = torch.tensor(labels)
    return img, labels

