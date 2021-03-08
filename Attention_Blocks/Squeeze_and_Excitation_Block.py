#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Squeeze_and_Excitation_Block.py
@Time    :   2021/03/05 23:00:02
@Author  :   Forskamse
@Version :   1.0
@Contact :   mb85453@@um.edu.mo
@License :   (C)Copyright 2019-2021, UM, SKL-IOTSC, MACAU, CHINA
@Desc    :   Tensorflow 2.x implementation of Squeeze-and-Excitation Block from the paper: Squeeze-and-Excitation Networks.
@Ref     :   [1]https://arxiv.org/abs/1709.01507
'''

import tensorflow as tf
from tensorflow.keras import layers, Sequential

class SqueezeExcitationBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, reduction_ratio = 16):
        super(SqueezeExcitationBlock, self).__init__()

        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        self.global_avgpool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(in_channels // reduction_ratio, activation='relu')
        self.dense2 = layers.Dense(in_channels, activation='sigmoid')
        
    def call(self, x):

        prev_x = x
        x = self.global_avgpool(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = tf.expand_dims(x, 1)
        x = prev_x * tf.expand_dims(x, 1)

        return x
    
    def get_config(self):
        base_config = super(SqueezeExcitationBlock, self).get_config()
        base_config['in_channels'] = self.in_channels
        base_config['reduction_ratio'] = self.reduction_ratio
        return base_config