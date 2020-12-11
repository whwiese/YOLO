# YOLO Object Detection

The YOLO (You Only Look Once) object detector was introduced in the paper [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) by Joseph Redmon et al. in 2015.

It is an efficient and powerful model built for real-time object detection, and has been updated many times since its introduction in 2015.
In this repository I build multiple object detection models based on the various versions of YOLO, and test them on the Pascal VOC object detection
dataset. The models are meant to be flexible, and I intend to make use of them in other object detection projects.

---
### YOLOv1: a short summary

Contrary to other early CNN-based object detection approaches, YOLO solves the object detection problem by feeding the entire input image
through a CNN only once (hence the name). The network itself divides the image into an SxS grid. Each grid space
contains B bounding boxes and its own outputs for classification among C classes. S, B and C are hyperparameters that the user can set based on their
problem and data.

The neural network itself contains 24 convolutional layers and two fully connected layers at the end. It also contains periodic max pooling layers, 
and while the original paper did not include batch normalization, later versions of YOLO do, and my implementation does. The output of the last 
fully connected layer is a (batch_size, S * S * (C+B * 5)) tensor, which is later reshaped to a (batch_size, S, S, C+B * 5) tensor for making predictions.
In this form each grid space contains B * 5 bounding box values ((x_mid, y_mid, width, height, probability) for each box) and C outputs for classification. 
Therefore each grid space can predict B different bounding boxes, but only one class shared by each of them.

The output predictions are processed with a probability threshold and non-max supression prior to evaluation. This should eliminate extraneous bounding boxes.

---

### Change Log

#### YOLOv2_lite (12/11/2020)

YOLOv2_lite is a new model located in model.py which incorporates the darknet-19 architecture used in YOLOv2. The primary purpose of YOLOv2-lite is to
reduce memory usage by getting rid of the fully-connected layers at the end of YOLOv1.

YOLOv2_lite differs from YOLOv2 in that it does not use a passthrough layer to help the network learn more effectively based on fine-grained features. 
The main complication with the passthrough layer is that it concatenates a 26x26 conv layer output to a 13x13 conv layer output, so it needs to be
implemented with a reorganization layer to reduce the width and height of the 26x26 component to 13x13 (quadrupling its depth in the process). I 
may implement this in the future if YOLOv2_lite performance is unsatisfactory.

Quick tests have shown you that YOLOv2_lite is able to overfit a training set of 100 images as well as YOLOv1 even without anchor boxes, and it runs
Thank slightly faster. More extensive testing is certainly required to compare the two models, but for now YOLOv2_lite appears to be an acceptable 
memory reduction solution.

##### Some results of overfitting 100 images for 100 epochs (0.768 mAP [0.05:0.95:0.05])

<img src="https://github.com/whwiese/YOLOv1/blob/master/predictions/2l_100e/Airplane3.png" alt="gen" width="400"/> <img 
src="https://github.com/whwiese/YOLOv1/blob/master/predictions/2l_100e/TableChairSofa.png" alt="gen" width="400"/>

---

### Guide to files

model.py: Contains models. Current modles are YOLOv1, YOLOv2_lite

train.py: Trains and optionally saves a YOLO model. set hyperparameters and data path using variables at the top of the file.

test_model.py: Runs the forward pass of a loaded model, computes mean average precision and plots results.

loss.py: The YOLO loss function as specified in the paper.

dataset.py: Creates PyTorch dataset object. Converts images and labels into YOLO form (SxS grid).

utils.py: Utility functions

---

### Get the data (Pascal VOC)

[Aladdin Persson's Kaggle repository](https://www.kaggle.com/dataset/734b7bcb7ef13a045cbdd007a3c19874c2586ed0b02b4afc86126e89d00af8d2)
 contains some nicely prepared csv files that can be used with this model along with the Pascal VOC images and labels. This is 
 an easy way to get the model up and running quickly.
 
[Or get the images and labels here and process them yourself](https://pjreddie.com/projects/pascal-voc-dataset-mirror/).

Or bring your own data! This implemetation of YOLO is designed to handle data with an arbitrary number of classes. 
Though some slight adjusting of dataset.py may be required if bounding boxes are not labeled in (x_mid, y_mid, width, height) form.

___

### Some predictions (after overfitting 100 training images for 26 epochs):
<img src="https://github.com/whwiese/YOLOv1/blob/master/predictions/Birds.png" alt="gen" width="400"/> <img 
src="https://github.com/whwiese/YOLOv1/blob/master/predictions/DiningRoom.png" alt="gen" width="400"/> <img 
src="https://github.com/whwiese/YOLOv1/blob/master/predictions/Sheep.png" alt="gen" width="400"/> <img 
src="https://github.com/whwiese/YOLOv1/blob/master/predictions/AirplanePerson2.png" alt="gen" width="400"/>

...something's being learned, but not very fast! Email me if you want to employ me or buy me a GPU (whwiese@berkeley.edu)

---

### References

[1] [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) - Joseph Redmon et al., 2015.

[2] [YOLO 9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) - Joseph Redmon, Ali Farhadi, 2016.

[3] [Aladdin Persson's YOLOv1 implementation](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO).

[4] [Aladdin Persson's Kaggle repository of processed Pascal VOC Data](https://www.kaggle.com/dataset/734b7bcb7ef13a045cbdd007a3c19874c2586ed0b02b4afc86126e89d00af8d2)

[5] [Aladdin Persson's mAP, nms, IoU test cases](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML_tests/Object_detection_tests)

[6] [Pascal VOC Dataset Mirror](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)
