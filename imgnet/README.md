# ImageNet Experiments

This directory contains code to train ResNet-152 models on ImageNet. Use the `train_152.py` script to perform training and `valIM152.py` script for validation (with 10-crop testing). You will need to download the ILSVRC clsloc dataset to a directory on your machine. 

Create files `train.txt` and `val.txt` in that directory where each line specifies the relative path to an image file and a class label (between 0 and 999), separated by a comma. e.g.,
```
train/n02397096/n02397096_10129.JPEG,343
```
