# SmileCNN

## Note
This repository is forked from [SmileCNN](https://github.com/kylemcdonald/SmileCNN) originally written by [kylemcdonald](https://github.com/kylemcdonald/).

Smile Detection with a Deep Convolutional Neural Net using [Keras-MXNet](https://github.com/awslabs/keras-apache-mxnet).
This example is based on the `mnist_cnn.py` example, running at 32x32 instead of 28x28.

This package contains 3 python files - `datasetprep.py`, `train.py` and `evaluation.py`
It also contains a folder - `keras-mms` which contains files for hosting this inference on MXNet Model Server


# Setup

Run `sudo python setup.py install` to add the necessary dependencies


# Documentation

Please follow instructions on the [medium blogpost](https://medium.com/apache-mxnet/deploy-a-smile-detector-with-keras-mxnet-and-mxnet-model-server-48cd9741b6d2) to use this smile detection model with MXNet Model Server
