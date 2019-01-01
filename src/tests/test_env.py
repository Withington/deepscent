#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
import tensorflow.keras as keras

def test_env():
    print('Python version is', sys.version_info)
    print('TensorFlow version is', tf.VERSION)
    print('TensorFlow Keras version is', tf.keras.__version__)
    assert tf.VERSION 
    assert tf.keras.__version__ 

def main():
    test_env()

if __name__ == "__main__":
    main()