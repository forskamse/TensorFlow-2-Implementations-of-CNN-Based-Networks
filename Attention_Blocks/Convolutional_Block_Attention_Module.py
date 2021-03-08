#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Convolutional_Block_Attention_Module.py
@Time    :   2021/03/05 22:58:58
@Author  :   Forskamse
@Version :   1.0
@Contact :   mb85453@@um.edu.mo
@License :   (C)Copyright 2019-2021, UM, SKL-IOTSC, MACAU, CHINA
@Desc    :   Tensorflow 2.x implementation of Convolutional Block Attention Module from the paper: CBAM: Convolutional Block Attention Module
@Ref     :   [1]https://github.com/kobiso/CBAM-tensorflow-slim/blob/master/nets/attention_module.py [2]https://arxiv.org/abs/1807.06521
'''

import tensorflow as tf
from tensorflow.keras import layers, Sequential

class ChannelAttentionModule(tf.keras.layers.Layer):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()

        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        self.avgpool1 = layers.GlobalAveragePooling2D()
        self.maxpool1 = layers.GlobalMaxPool2D()
        
        # Shared MLP
        self.dense1 = layers.Dense(in_channels // reduction_ratio, activation='relu')
        self.dense2 = layers.Dense(in_channels, activation='relu')

        self.ac_channel = layers.Activation('sigmoid')
        self.reshape_channel = layers.Reshape((1, 1, in_channels))
    
    def call(self, x):
        
        # Channel Attention
        avgpool = self.avgpool1(x)
        maxpool = self.maxpool1(x)
        avg_out = self.dense2(self.dense1(avgpool))
        max_out = self.dense2(self.dense1(maxpool))
        channel_out = layers.add([avg_out, max_out])
        channel_out = self.ac_channel(channel_out)
        channel_out = self.reshape_channel(channel_out)
        
        return channel_out
    
    def get_config(self):
        base_config = super(ChannelAttentionModule, self).get_config()
        base_config['in_channels'] = self.in_channels
        base_config['reduction_ratio'] = self.reduction_ratio
        return base_config


class SpatialAttentionModule(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()

        self.conv_spatial = layers.Conv2D(1, 7, padding='same')
        self.ac_spatial = layers.Activation('sigmoid')
    
    def call(self, x):

        # Spatial Attention
        avgpool = tf.reduce_mean(x, axis=3, keepdims=True)
        maxpool = tf.reduce_max(x, axis=3, keepdims=True)
        spatial = layers.Concatenate(axis=3)([avgpool, maxpool])
        spatial = self.conv_spatial(spatial)
        spatial_out = self.ac_spatial(spatial)
        
        return spatial_out

    def get_config(self):
        base_config = super(SpatialAttentionModule, self).get_config()
        return base_config


class CBAM(tf.keras.layers.Layer):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        self.cam = ChannelAttentionModule(in_channels, reduction_ratio)
        self.sam = SpatialAttentionModule()
    
    def call(self, x):

        x = self.cam(x) * x
        x = self.sam(x) * x

        return x
    
    def get_config(self):
        base_config = super(CBAM, self).get_config()
        base_config['in_channels'] = self.in_channels
        base_config['reduction_ratio'] = self.reduction_ratio
        return base_config