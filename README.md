
# Stereo Depth Estimation using Machine Learning

Depth estimation is a trivial concept for humans but a difficult task for machines. Understanding of depth is at the core of applications like 3D Scene construction, Augmented Reality and Autonomous
Driving. Here, in this project I present a deep learning approach using 2D CNN and 3D CNN
architecture to replicate the ground truth disparity of the stereo images.


## Machine Learning Model
This model consists of a Two-Dimensional Convolutional
Neural Architecture followed by a Three-Dimensional Convolutional Neural Architecture. The output from this deep learning model is finally passed on to a
Disparity Regressor. The input consists of left and right images from the dataset
passed to the 2D-CNN one after another thus, utilizing weight-sharing. Now, we
get the output features from this 2D-CNN for both left and right images which
in turn is now combined to form a Cost Volume Matrix. This new cost volume
is passed to the next 3D-CNN and the output from this CNN is passed on to
the regressor to get our final output image.

[![upload-github1.jpg](https://i.postimg.cc/FRS2JJCw/upload-github1.jpg)](https://postimg.cc/47J8Dng1)

## Dataset
Stereo Depth Estimation dataset: KITI and the sceneflow dataset
[![upload-github4.jpg](https://i.postimg.cc/T18b0hjz/upload-github4.jpg)](https://postimg.cc/k6cDXJps)

## Table depicting the 2D CNN used
[![upload-github2.jpg](https://i.postimg.cc/pLFXRkDt/upload-github2.jpg)](https://postimg.cc/LJmMVtCy)

## Table depicting the 3D CNN used
[![upload-github3.jpg](https://i.postimg.cc/bw70w6W0/upload-github3.jpg)](https://postimg.cc/SnGYZGyj)
## Sample Images generated

[![r5.png](https://i.postimg.cc/nhRb8xQS/r5.png)](https://postimg.cc/68v1RDcC)

[![r6.png](https://i.postimg.cc/YSZ59xR1/r6.png)](https://postimg.cc/SXLT1Cws)

## Testing phase 

The model shown in tables 2 and 3 is the final model with 4730081 parameters
that was initially run for 100 epochs on the KITI dataset but as mentioned above
the dataset was replicated three times with data augmentation. Each epoch was
taking close to 13 minutes to run and after 100 epochs, the original dataset
without any augmentation was passed to the model for another 150 epochs i.e.
only the first 150 training images and loss was computed on them.

## Final Images generated from the model

[![r1.png](https://i.postimg.cc/zvZYdkRJ/r1.png)](https://postimg.cc/xJRZ8MR7)

[![r2.png](https://i.postimg.cc/C5GT9fjb/r2.png)](https://postimg.cc/q6v53qD7)

[![r3.png](https://i.postimg.cc/XNdbrtTX/r3.png)](https://postimg.cc/RqS88pCz)

[![r4.png](https://i.postimg.cc/BQ3GHM0p/r4.png)](https://postimg.cc/Z0wghP0v)

## Conclusion
Utilizing deep learning in the field of depth estimation provides an alternative
to traditional technique and expensive equipment and gives nice results as well.
In this project, even though the losses obtained through the model were not
as good but the model is learning to understand the concept at hand. Maybe,
Pre-training would have solved this problem as the dataset size was too small.
However, swithching to pretrained models like Resnet, VGG and r3d_18
helped in designing better model architecture than starting from scratch.
## Authors

- [@Pahulmeet](https://github.com/Pahulmeet)

