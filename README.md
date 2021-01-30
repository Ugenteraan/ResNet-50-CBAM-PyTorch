# ResNet-50 with CBAM using PyTorch 1.8

### Introduction
This repository contains the implementation of ResNet-50 with and without CBAM. Note that some parameters of the architecture may vary such as the kernel size or strides of convolutional layers. The implementation was tested on Intel's Image Classification dataset that can be found [here](https://www.kaggle.com/puneet6060/intel-image-classification). 

### Performance
ResNet-50 **with** CBAM achieved an accuracy of **86.6%** on the validation set while ResNet-50 **without** CBAM achieved an accuracy of **84.34%** on the same validation set. The figures below show the improvement of the models over the epochs.

<div align="center"> 
ResNet-50 Without CBAM 
</div>


