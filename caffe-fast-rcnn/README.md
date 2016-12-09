## Caffe for PVANET
by Kye-Hyeon Kim, Yeongjae Cheon, Sanghoon Hong, Byungseok Roh, Minje Park (Intel Imaging and Camera Technology)

### Introduction

This repository is a fork from [BVLC/caffe](https://github.com/BVLC/caffe). Some modifications have been made to run PVANET with Caffe:
- Implemented a new learning rate scheduling based on plateau detection.
- Implemented proposal layer for both CPU and GPU versions.
- Implemented NMS for both CPU and GPU versions.
- Copied RoI pooling layer and smoothed L1 loss layer from [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)

NOTE: For running PVANET, cuDNN (v5 for CUDA 7.5) is significantly slower than Caffe's native implementation.
