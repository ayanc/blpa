# CIFAR Experiments

This directory contains code to train ResNet-164 models on CIFAR-10 and CIFAR-100. Use the `train_res.py` script to perform training. The model description is in `ResNet.py`. You will need to download CIFAR data to `data/cifar100*` and `data/cifar10/*` directories respectively: use the `download_*.sh` scripts for this.

To get test set numbers after, call `train_res.py` with `-bsz 500` (to get more accurate batch statistics).
