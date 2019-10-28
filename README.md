# Backprop with Low-Precision Activations

This distribution provides a library implementing the approximate memory-efficient deep network training algorithm described in our paper:

Ayan Chakrabarti and Benjamin Moseley, "**[Backprop with Approximate Activations for Memory-efficient Network Training](https://arxiv.org/abs/1901.07988)**", NeurIPS 2019.


If you find our method or library useful in your research, please consider citing the above paper.

## Installation

The library is implemented over Tensorflow, and you will need Tensorflow installed before using the library. You will also need to compile the quantization ops in `blpa/ops/`. Specifically, you need to compile `quant.cc.` and `quant.cc.cu` into a loadable library. A sample compilation script is provided as `make.sh`, which might work for you out of the box. If it doesn't, please look at the instructions at [https://www.tensorflow.org/guide/extend/op](https://www.tensorflow.org/guide/extend/op).

Also note that the python code assumes that your library will be compiled as `quant.so`, which is the case on Linux systems. On other systems, the loadable library may have a different extension. If that is the case, please modify the `sopath` definition at the top of `blpa/quant.py` accordingly.

## Usage

The `blpa.graph` and `blpa.mgraph` modules provide routines for defining and training Residual network models on a single, and multiple GPUs respectively. Broadly, network models are defined in terms of a sequence of layers, with specifications for non-linearities (pooling, ReLU, batch-norm, etc.) and residual connections to apply *prior* to each (i.e., to the layer's inputs). Note that the library only supports residual networks with a "width" of 2: i.e., there can be upto one outstanding residual connection at a time whose value is stored in a residual buffer. For each layer, you specify whether to add the contents of the buffer to the current layer's inputs, and/or to copy this input (after addition, and before or after the non-linearities) to the buffer.

Please see the sample model and training code in the `cifar/` and `imagenet/` directories for example usage. These also specify how to use the available simple data layer (cifar), or to specify custom data layers (imagenet) as Tensorflow graphs.  

Contact ayan@wustl.edu with any questions.
