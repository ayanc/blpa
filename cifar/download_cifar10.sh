#!/bin/bash

mkdir -p data/cifar10
cd data/cifar10
curl https://projects.ayanc.org/blpa/cf10/test.npz > test.npz
curl https://projects.ayanc.org/blpa/cf10/train.npz.aa https://projects.ayanc.org/blpa/cf10/train.npz.ab https://projects.ayanc.org/blpa/cf10/train.npz.ac https://projects.ayanc.org/blpa/cf10/train.npz.ad https://projects.ayanc.org/blpa/cf10/train.npz.ae  > train.npz
