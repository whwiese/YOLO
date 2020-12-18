import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import (YOLOv1, YOLOv2_lite)
from dataset import YOLOVOCDataset
from utils import (
        plot_detections,
        mean_average_precision,
        get_bboxes,
)

#YOLO HYPERPARAMETERS
GRID_SIZE = 13
NUM_BOXES = 2
NUM_CLASSES = 20

DATA_CSV = "data/100examples.csv"
IMG_DIR = "data/images"
LABEL_DIR= "data/labels"
MODEL_PATH = "saved_models/overfit_100_2l_100e.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, labels):
        for t in self.transforms:
            img, labels = t(img), labels

        return img, labels

transform = Compose([transforms.Resize((416,416)), transforms.ToTensor()])

def main():

    model = YOLOv2_lite(grid_size=GRID_SIZE, num_boxes=NUM_BOXES, 
            num_classes=NUM_CLASSES).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    test_dataset = YOLOVOCDataset(
            DATA_CSV, transform=transform, img_dir=IMG_DIR,
            label_dir=LABEL_DIR, S=GRID_SIZE, B=NUM_BOXES,
            C=NUM_CLASSES, hflip_prob=0.5,
    )

    test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=False,
    )

    images, predictions, labels = get_bboxes(
            test_loader, model, iou_threshold=0.5, prob_threshold=0.4,
            S=GRID_SIZE, C=NUM_CLASSES, mode="all",
            return_images=True,
    )
    
    mAP = mean_average_precision(predictions, labels,
            num_classes=NUM_CLASSES) 

    print("Mean Average Precision: %.3f"%(mAP))

    img_to_plot = np.random.randint(0,len(images))
    
    image = images[img_to_plot]

    img_preds = [pred[1:] for pred in predictions
                    if pred[0] == img_to_plot
                ]
    img_labels = [label[1:] for label in labels
                    if label[0] == img_to_plot
                ]

    #plot_detections(image, img_preds) 
    plot_detections(image, img_labels) 

if __name__ == "__main__":
    main()
