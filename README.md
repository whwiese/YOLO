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

### Update Log

#### I found a GPU / Better living with data augmentation (12/24/2020)

A MacBook's CPU can only deliver so much when it comes to deep learning, so I turned to Google Colab to allow me to train this thing for real. I found that even on a training set of ~16k images my YOLOv1 and v2_lite models overfit the training data and performed extremely poorly on the held-out validation set. Fortunately, through the wonders of data augmentation I was able to train a YOLOv1 model that performed well on the validation set, and on images outside of Pascal VOC entirely.

##### Data Augmentation Techniques

Horizontal Flip: With tunable probability p images are flipped horizontally, effectively doubling the amount of training images we have. I had to implement this as a function of my custom dataset class in PyTorch so the ground truth bounding box could be correctly transformed along with the image.

Random Crop: Given a crop factor cf and an input image of size (x,y), the image is resized to a random value between (x,y) and (x*(1+cf), y*(1+cf)), then cropped to size (x_out,y_out) by shaving off a random corner. This helps tremendously with overfitting, as the ground truth bounding boxes for a given image will have different values almost every time the model sees the image. This prevents the model from solving the training set by recognizing some unique feature of the image and mapping to the unchanging ground truth bounding box coordinates. Random cropping is the big change that took this model from simple overfitting to one that learns general features and can take on images not seen in the training set. Like horizontal flip, this had to be implemented as a function of the custom dataset class.

Color Jitter and Gaussian Blur: Color jitter and gaussian blur on test images help the model become more robust and deal with images outside of the PASCAL VOC dataset. Gaussian blur specifically provided an noicable increase in performance on videos I took on my iPhone. Since these transformations leave the ground truth bounding boxes unchanged I was able to implement them with PyTorch's built-in transformations.

##### PASCAL VOC Validation Set Results (Trained for 200 epochs on a random 80% of Pascal VOC)

The model is quite good at classification and bounding box prediction for images with a single prominent object.

<img src="https://github.com/whwiese/YOLOv1/blob/master/predictions/v1_hrc_200e/good/Dog.png" alt="gen" width="400"/> <img 
src="https://github.com/whwiese/YOLOv1/blob/master/predictions/v1_hrc_200e/good/Car.png" alt="gen" width="400"/>

But things get a little dicey as images get busier.

<img src="https://github.com/whwiese/YOLOv1/blob/master/predictions/v1_hrc_200e/Meh/IndoorCar.png" alt="gen" width="400"/> <img 
src="https://github.com/whwiese/YOLOv1/blob/master/predictions/v1_hrc_200e/Meh/LittleBike.png" alt="gen" width="400"/>

And there are occasional obvious classification failures. This is probably due to the limited dataset, as my data augmentation techniques are more helpful for bounding box predictions than for classification.


<img src="https://github.com/whwiese/YOLOv1/blob/master/predictions/v1_hrc_200e/bad/NotCat.png" alt="gen" width="400"/> <img 
src="https://github.com/whwiese/YOLOv1/blob/master/predictions/v1_hrc_200e/bad/BigBird.png" alt="gen" width="400"/>

For more validation set predictions check out the predictions/v1_hrc_200e folder in this repository. There are some pretty interesting results in there.


#### YOLOv2_lite (12/11/2020)

YOLOv2_lite is a new model located in model.py which incorporates the darknet-19 architecture used in YOLOv2. The primary purpose of YOLOv2_lite is to
reduce memory usage by getting rid of the fully-connected layers at the end of YOLOv1.

YOLOv2_lite differs from YOLOv2 in that it does not use a passthrough layer to help the network learn more effectively based on fine-grained features. 
The main complication with the passthrough layer is that it concatenates a 26x26 conv layer output to a 13x13 conv layer output, so it needs to be
implemented with a reorganization layer to reduce the width and height of the 26x26 component to 13x13 (quadrupling its depth in the process). I 
may implement this in the future if YOLOv2_lite performance is unsatisfactory.

Quick tests have shown you that YOLOv2_lite is able to overfit a training set of 100 images as well as YOLOv1 even without anchor boxes, and it runs
Thank slightly faster. More extensive testing is certainly required to compare the two models, but for now YOLOv2_lite appears to be an acceptable 
memory reduction solution.

##### Some results of overfitting 100 images for 100 epochs (0.768 mAP [0.05:0.95:0.05])

<img src="https://github.com/whwiese/YOLOv1/blob/master/predictions/v1_hrc_200e/Dog.png" alt="gen" width="400"/> <img 
src="https://github.com/whwiese/YOLOv1/blob/master/predictions/v1_hrc_200e/Car.png" alt="gen" width="400"/>

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
