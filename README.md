# YOLOv1 Object Detection

The YOLO (You Only Look Once) object detector was introduced in the paper [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) by Joseph Redmon et al. in 2015.

This project was heavily influenced by 
[Aladdin Persson's YOLOv1 implementation](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO).
I made some different design choices and generalized the model by allowing the user to easily adjust the YOLO hyperparameters (grid size, number of 
bounding boxes per grid space, number of classes for classification problem) in train.py, but the structure of the project came from Persson's implementation. 
I also used data files from [Persson's Kaggle repository](https://www.kaggle.com/dataset/734b7bcb7ef13a045cbdd007a3c19874c2586ed0b02b4afc86126e89d00af8d2)
to train and test the model.

---
#### YOLOv1: a short summary

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

#### Guide to files

model.py: Contains the neural network.

train.py: Trains and optionally saves a YOLO model. set hyperparameters and data path using variables at the top of the file.

test_model.py: Runs the forward pass of a loaded model, computes mean average precision and plots results.

loss.py: The YOLO loss function as specified in the paper.

dataset.py: Creates PyTorch dataset object. Converts images and labels into YOLO form (SxS grid).

utils.py: Utility functions

---

#### Some Predictions (after overfitting 100 training images for 26 epochs):
<img src="https://github.com/whwiese/YOLOv1/blob/master/predictions/Birds.png" alt="gen" width="400"/> <img 
src="https://github.com/whwiese/YOLOv1/blob/master/predictions/DiningRoom.png" alt="gen" width="400"/> <img 
src="https://github.com/whwiese/YOLOv1/blob/master/predictions/Sheep.png" alt="gen" width="400"/> <img 
src="https://github.com/whwiese/YOLOv1/blob/master/predictions/AirplanePerson2.png" alt="gen" width="400"/>

...something's being learned but not very fast! email me if you want to buy me a GPU (whwiese@berkeley.edu)
