import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from torch.utils.data import (DataLoader, random_split)
from model import (YOLOv1, YOLOv2_lite)
from dataset import YOLOVOCDataset
from utils import (
        intersection_over_union,
        non_max_supression,
        mean_average_precision,
        single_map,
        get_bboxes,
)
from loss import YoloLoss
import time
import pandas as pd

#YOLO hyperparameters
GRID_SIZE = 7
NUM_BOXES = 2
NUM_CLASSES = 20

#other hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 24
WEIGHT_DECAY = 0
EPOCHS = 200
NUM_WORKERS = 24
PIN_MEMORY = True
LOAD_CHECKPOINT = True
DROP_LAST = True
TRAINING_DATA = "all_voc.csv"
TEST_DATA = None 
SAVE_CHECKPOINT_PATH = "VOCTrial_v1_hflip3/"
LOAD_CHECKPOINT_PATH = "VOCTrial_v1_hflip3/checkpoint_180e.pt"
IMG_DIR = "images"
LABEL_DIR = "labels"

DATA_SIZE = len(pd.read_csv(TRAINING_DATA))
TRAIN_SIZE = int(0.8*DATA_SIZE)
VAL_SIZE = DATA_SIZE-TRAIN_SIZE

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, labels):
        for t in self.transforms:
            img, labels = t(img), labels

        return img, labels

transform = Compose([transforms.Resize((448,448)), transforms.ToTensor()])

train_transform = Compose([transforms.Resize((448,448)),
                           transforms.ColorJitter(brightness=0.5, contrast = 0.5),
                           transforms.ToTensor(),
            ])

def train_fn(loader_train, model, optimizer, loss_fn):
    
    mean_loss = []

    for batch_index, (x, y) in enumerate(loader_train):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Mean loss: %f"%(sum(mean_loss)/len(mean_loss)))

def main():
    model = YOLOv1(grid_size=GRID_SIZE, 
            num_boxes=NUM_BOXES, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss(S=GRID_SIZE, B=NUM_BOXES, C=NUM_CLASSES)

    epochs_passed = 0

    if LOAD_CHECKPOINT:
        checkpoint = torch.load(LOAD_CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in optimizer.param_groups:
          param_group['lr'] = 2e-6
        epochs_passed = checkpoint['epoch']

    mask_dataset = YOLOVOCDataset(
            TRAINING_DATA, transform=train_transform, 
            img_dir=IMG_DIR, label_dir=LABEL_DIR,
            S=GRID_SIZE, B=NUM_BOXES, C=NUM_CLASSES,
            hflip_prob = 0.5,
            random_crops=0.25,
    )

    train_dataset, val_dataset = random_split (
            mask_dataset, [TRAIN_SIZE, VAL_SIZE], 
            generator=torch.Generator().manual_seed(42),
    )

    if torch.cuda.is_available():
        train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                shuffle=True,
                drop_last=DROP_LAST,
        )
        
        val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                shuffle=True,
                drop_last=False,
        )

    else:
        train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=BATCH_SIZE,
                #num_workers=NUM_WORKERS,
                #pin_memory=PIN_MEMORY,
                shuffle=True,
                drop_last=DROP_LAST,
        )
        
        val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=BATCH_SIZE,
                #num_workers=NUM_WORKERS,
                #pin_memory=PIN_MEMORY,
                shuffle=True,
                drop_last=False,
        )

    for epoch in range(EPOCHS):
        if epochs_passed%2 == 0:
            train_pred_boxes, train_target_boxes = get_bboxes(
                    train_loader, model, iou_threshold=0.35, prob_threshold=0.4,
                    S=GRID_SIZE, C=NUM_CLASSES, mode = "batch",
                    device=DEVICE,
            )
            val_pred_boxes, val_target_boxes = get_bboxes(
                    val_loader, model, iou_threshold=0.35, prob_threshold=0.4,
                    S=GRID_SIZE, C=NUM_CLASSES, mode = "batch",
                    device=DEVICE,
            )

            # map function takes predicted boxes and ground truth
            # boxes in form [[],[],[],...] where each sublist is a bounding box
            # of form [image_index, class_pred, x_mid, y_mid, w, h, prob]
            train_mAP = single_map(
                    train_pred_boxes, train_target_boxes, 
                    box_format="midpoint", num_classes=NUM_CLASSES,
                    iou_threshold=0.35,
            )
            val_mAP = single_map(
                    val_pred_boxes, val_target_boxes, 
                    box_format="midpoint", num_classes=NUM_CLASSES,
                    iou_threshold=0.35,
            )

            train_mAPs.append(train_mAP)
            val_mAPs.append(val_mAP)
            epochs_recorded.append(epochs_passed)

            print("Train mAP: %f"%(train_mAP))
            print("Val mAP: %f"%(val_mAP))

        if epochs_passed%10 == 0:
            save_path = SAVE_CHECKPOINT_PATH + ("checkpoint_%de"%(epochs_passed))+".pt"
            checkpoint = { 
                'epoch': epochs_passed,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint,save_path)
            print("Trained for %d epochs"%(epochs_passed))

        train_fn(train_loader, model, optimizer, loss_fn)
        epochs_passed += 1

train_mAPs = []
val_mAPs = []
epochs_recorded = []

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Training time: %f seconds"%(time.time()-start_time))
