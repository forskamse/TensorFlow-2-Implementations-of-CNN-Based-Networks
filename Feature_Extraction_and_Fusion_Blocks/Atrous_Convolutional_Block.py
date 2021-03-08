#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Atrous_Convolutional_Block.py
@Time    :   2021/03/05 22:20:38
@Author  :   Forskamse
@Version :   1.0
@Contact :   mb85453@@um.edu.mo
@License :   (C)Copyright 2019-2021, UM, SKL-IOTSC, MACAU, CHINA
@Desc    :   Tensorflow 2.x implementation of Atrous Convolutional Blocks for 1D and 2D inputs, suggested by the paper: An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
@Ref     :   [1]https://github.com/philipperemy/keras-tcn [2]https://github.com/Baichenjia/Tensorflow-TCN/blob/master/tcn.py [3]https://arxiv.org/pdf/1803.01271.pdf
'''

import tensorflow as tf
from tensorflow.keras import layers, Sequential


class AtrousConvBlock_1D(tf.keras.layers.Layer):
    def __init__(self, dilation_rate, inter_channels, kernel_size, padding, dropout_rate): 
        super(AtrousConvBlock_1D, self).__init__()
        assert padding in ['causal', 'same']

        self.conv1 = layers.Conv2D(inter_channels, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding)
        self.batch1 = layers.BatchNormalization(axis=-1)
        self.ac1 = layers.Activation('relu')
        self.drop1 = layers.Dropout(rate=dropout_rate)

        self.conv2 = layers.Conv2D(inter_channels, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding)
        self.batch2 = layers.BatchNormalization(axis=-1)		
        self.ac2 = layers.Activation('relu')
        self.drop2 = layers.Dropout(rate=dropout_rate)

        self.downsample = layers.Conv2D(inter_channels, kernel_size=1, padding='same')
        self.ac3 = layers.Activation('relu')


    def call(self, x, training):
        prev_x = x
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.ac1(x)
        x = self.drop1(x) if training else x

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.ac2(x)
        x = self.drop2(x) if training else x

        if prev_x.shape[-1] != x.shape[-1]:
            prev_x = self.downsample(prev_x)

        return self.ac3(prev_x + x)

class AtrousConvBlock_2D(tf.keras.layers.Layer):
    def __init__(self, dilation_rate, inter_channels, kernel_size, dropout_rate, padding): 
        super(AtrousConvBlock_2D, self).__init__()
        assert padding in ['same']
        
        self.conv1 = layers.Conv2D(inter_channels, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding)
        self.batch1 = layers.BatchNormalization(axis=-1)
        self.ac1 = layers.Activation('relu')
        self.drop1 = layers.Dropout(rate=dropout_rate)

        self.conv2 = layers.Conv2D(inter_channels, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding)
        self.batch2 = layers.BatchNormalization(axis=-1)		
        self.ac2 = layers.Activation('relu')
        self.drop2 = layers.Dropout(rate=dropout_rate)

        self.downsample = layers.Conv2D(inter_channels, kernel_size=1, padding='same')
        self.ac3 = layers.Activation('relu')


    def call(self, x, training):
        prev_x = x
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.ac1(x)
        x = self.drop1(x) if training else x

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.ac2(x)
        x = self.drop2(x) if training else x

        if prev_x.shape[-1] != x.shape[-1]:
            prev_x = self.downsample(prev_x)

        return self.ac3(prev_x + x)