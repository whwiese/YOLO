import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader
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

seed = 123
torch.manual_seed(seed)

#YOLO hyperparameters
GRID_SIZE = 13
NUM_BOXES = 2
NUM_CLASSES = 20

#other hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 10
WEIGHT_DECAY = 0
EPOCHS = 200
NUM_WORKERS = 2
PIN_MEMORY = False
LOAD_MODEL = True
DROP_LAST = False
TRAINING_DATA = "data/100examples.csv"
TEST_DATA = "data/100examples.csv"
SAVE_MODEL_PATH = "saved_models/2l_overfit_100_2l_100e.pt"
LOAD_MODEL_PATH = "saved_models/2l_overfit_100.pt"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, labels):
        for t in self.transforms:
            img, labels = t(img), labels

        return img, labels

transform = Compose([transforms.Resize((416,416)), transforms.ToTensor()])

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
    model = YOLOv2_lite(grid_size=GRID_SIZE, 
            num_boxes=NUM_BOXES, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss(S=GRID_SIZE, B=NUM_BOXES, C=NUM_CLASSES)

    if LOAD_MODEL:
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))

    train_dataset = YOLOVOCDataset(
            TRAINING_DATA, transform=transform, 
            img_dir=IMG_DIR, label_dir=LABEL_DIR,
            S=GRID_SIZE, B=NUM_BOXES, C=NUM_CLASSES,
    )
    
    test_dataset = YOLOVOCDataset(
            TEST_DATA, transform=transform, 
            img_dir=IMG_DIR, label_dir=LABEL_DIR,
            S=GRID_SIZE, B=NUM_BOXES, C=NUM_CLASSES,
    )
   
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            #num_workers=NUM_WORKERS,
            #pin_memory=PIN_MEMORY,
            shuffle=True,
            drop_last=False,
    )
    
    test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            #num_workers=NUM_WORKERS,
            #pin_memory=PIN_MEMORY,
            shuffle=True,
            drop_last=False,
    )

    epochs_passed = 0

    for epoch in range(EPOCHS):
        pred_boxes, target_boxes = get_bboxes(
                train_loader, model, iou_threshold=0.5, prob_threshold=0.4,
                S=GRID_SIZE, C=NUM_CLASSES, mode = "batch",
        )
        # map function takes predicted boxes and ground truth
        # boxes in form [[],[],[],...] where each sublist is a bounding box
        # of form [image_index, class_pred, x_mid, y_mid, w, h, prob]

        mAP = single_map(
                pred_boxes, target_boxes, 
                box_format="midpoint", num_classes=NUM_CLASSES,
        )

        print("Train mAP: %f"%(mAP))
        if epochs_passed == 50:
            torch.save(model.state_dict(),SAVE_MODEL_PATH)
            print("Trained for %d epochs"%(epochs_passed))
            break

        train_fn(train_loader, model, optimizer, loss_fn)
        epochs_passed += 1

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Training time: %f seconds"%(time.time()-start_time))
