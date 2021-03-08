#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Receptive_Field_Block.py
@Time    :   2021/03/07 00:52:06
@Author  :   Forskamse
@Version :   1.0
@Contact :   mb85453@@um.edu.mo
@License :   (C)Copyright 2019-2021, UM, SKL-IOTSC, MACAU, CHINA
@Desc    :   Tensorflow 2.x implementation of Receptive Field Block, from the paper: Receptive Field Block Net for Accurate and Fast Object Detection
@Ref     :   [1]https://arxiv.org/pdf/1711.07767.pdf
'''

import tensorflow as tf
from tensorflow.keras import layers, Sequential

class ReceptiveFieldBlock(tf.keras.layers.Layer):
    def __init__(self, inter_channels, out_channels):

        self.inter_channels = inter_channels
        self.out_channels = out_channels

        super(ReceptiveFieldBlock, self).__init__()

        self.br1_conv1 = layers.Conv2D(inter_channels, kernel_size=(1, 1), padding='same')
        self.br1_bn1 = layers.BatchNormalization(axis=-1)
        self.br1_ac1 = layers.Activation(activation='relu')
        self.br1_conv2 = layers.Conv2D(inter_channels, kernel_size=(3, 3), dilation_rate=1, padding='same')
        self.br1_bn2 = layers.BatchNormalization(axis=-1)
        self.br1_ac2 = layers.Activation(activation='relu')

        self.br2_conv1 = layers.Conv2D(inter_channels, kernel_size=(1, 1), padding='same')
        self.br2_bn1 = layers.BatchNormalization(axis=-1)
        self.br2_ac1 = layers.Activation(activation='relu')
        self.br2_conv2 = layers.Conv2D(inter_channels, kernel_size=(3, 3), padding='same')
        self.br2_bn2 = layers.BatchNormalization(axis=-1)
        self.br2_ac2 = layers.Activation(activation='relu')
        self.br2_conv3 = layers.Conv2D(inter_channels, kernel_size=(3, 3), dilation_rate=3, padding='same')
        self.br2_bn3 = layers.BatchNormalization(axis=-1)
        self.br2_ac3 = layers.Activation(activation='relu')

        self.br3_conv1 = layers.Conv2D(inter_channels, kernel_size=(1, 1), padding='same')
        self.br3_bn1 = layers.BatchNormalization(axis=-1)
        self.br3_ac1 = layers.Activation(activation='relu')
        self.br3_conv2 = layers.Conv2D(inter_channels, kernel_size=(5, 5), padding='same')
        self.br3_bn2 = layers.BatchNormalization(axis=-1)
        self.br3_ac2 = layers.Activation(activation='relu')
        self.br3_conv3 = layers.Conv2D(inter_channels, kernel_size=(3, 3), dilation_rate=5, padding='same')
        self.br3_bn3 = layers.BatchNormalization(axis=-1)
        self.br3_ac3 = layers.Activation(activation='relu')
    
        self.conv_match = layers.Conv2D(out_channels, kernel_size=(1, 1), padding='same')
        self.bn_match = layers.BatchNormalization(axis=-1)

        self.conv_res = layers.Conv2D(out_channels, kernel_size=(1, 1), padding='same')
        self.bn_res = layers.BatchNormalization(axis=-1)

        self.ac_res = layers.Activation(activation='relu')

    def call(self, x):
        prev_x = x
        
        h = self.br1_bn1(x)
        h = self.br1_ac1(h)
        h = self.br1_conv1(h)
        h = self.br1_bn2(h)
        h = self.br1_ac2(h)
        x_br1 = self.br1_conv2(h)

        h = self.br2_bn1(x)
        h = self.br2_ac1(h)
        h = self.br2_conv1(h)       
        h = self.br2_bn2(h)
        h = self.br2_ac2(h)
        h = self.br2_conv2(h)
        h = self.br2_bn3(h)
        h = self.br2_ac3(h)
        x_br2 = self.br2_conv3(h)

        h = self.br3_bn1(x)
        h = self.br3_ac1(h)
        h = self.br3_conv1(h)
        h = self.br3_bn2(h)
        h = self.br3_ac2(h)
        h = self.br3_conv2(h)
        h = self.br3_bn3(h)
        h = self.br3_ac3(h)
        x_br3 = self.br3_conv3(h)

        x = layers.Concatenate(axis=-1)([x_br1, x_br2, x_br3])
        x = self.conv_match(x)
        x = self.bn_match(x)

        if prev_x.shape[-1] != x.shape[-1]:
            prev_x = self.conv_res(prev_x)
            prev_x = self.bn_res(prev_x)

        x = layers.add([x, prev_x])
        x = self.ac_res(x)
        return x

    def get_config(self):
        base_config = super(ReceptiveFieldBlock, self).get_config()
        base_config['inter_channels'] = self.inter_channels
        base_config['out_channels'] = self.out_channels
        return base_config