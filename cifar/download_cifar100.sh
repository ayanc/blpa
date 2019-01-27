#!/bin/bash

mkdir -p data/cifar100
cd data/cifar100
curl https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz | tar xzvf -
mv cifar-100-python/* .
rmdir cifar-100-python
