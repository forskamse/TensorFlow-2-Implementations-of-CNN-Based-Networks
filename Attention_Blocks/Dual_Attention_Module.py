#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Dual_Attention_Module.py
@Time    :   2021/03/05 17:35:59
@Author  :   Forskamse
@Version :   1.0
@Contact :   mb85453@@um.edu.mo
@License :   (C)Copyright 2019-2021, UM, SKL-IOTSC, MACAU, CHINA
@Desc    :   Tensorflow 2.x implementation of Dual Attention Module from the paper: Dual Attention Network for Scene Segmentation.
@Ref     :   [1]https://arxiv.org/pdf/1809.02983.pdf [2]https://github.com/niecongchong/DANet-keras/blob/master/layers/attention.py
'''

import tensorflow as tf
from tensorflow.keras import layers, Sequential

class ChannelAttentionModule(tf.keras.layers.Layer):
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None
                 ):
        super(ChannelAttentionModule, self).__init__()
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer=self.gamma_initializer,
                                     name='beta',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        input_shape = input.get_shape().as_list()
        _, h, w, filters = input_shape
        vec_a = layers.Reshape((h * w, filters))(input)
        aa = layers.dot([vec_a, vec_a], axes=1)
        softmax_aa = layers.Activation('softmax')(aa)
        aaa = layers.dot([vec_a, softmax_aa], axes=[2, 1])
        aaa = layers.Reshape((h, w, filters))(aaa)
        out = self.gamma*aaa + input
        return out

    def get_config(self):
        base_config = super(ChannelAttentionModule, self).get_config()
        base_config['gamma_initializer'] = self.gamma_initializer
        base_config['gamma_regularizer'] = self.gamma_regularizer
        base_config['gamma_constraint'] = self.gamma_constraint
        return base_config


class PositionAttentionModule(tf.keras.layers.Layer):
    def __init__(self,
                 in_channels,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None
                 ):
        super(PositionAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

        self.conv1 = layers.Conv2D(in_channels // 8, 1, use_bias=False, kernel_initializer='he_normal')
        self.conv2 = layers.Conv2D(in_channels // 8, 1, use_bias=False, kernel_initializer='he_normal')
        self.conv3 = layers.Conv2D(in_channels, 1, use_bias=False, kernel_initializer='he_normal')

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer=self.gamma_initializer,
                                     name='alpha',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        input_shape = input.get_shape().as_list()
        _, h, w, filters = input_shape

        b = self.conv1(input)
        c = self.conv2(input)
        d = self.conv3(input)

        vec_b = layers.Reshape((h * w, filters // 8))(b)
        vec_c = layers.Reshape((h * w, filters // 8))(c)
        bc = layers.dot([vec_b, vec_c], axes=2)
        softmax_bc = layers.Activation('softmax')(bc)
        vec_d = layers.Reshape((h * w, filters))(d)
        bcd = layers.dot([vec_d, softmax_bc], axes=[1, 2])
        bcd = layers.Reshape((h, w, filters))(bcd)
        out = self.gamma*bcd + input
        return out

    def get_config(self):
        base_config = super(PositionAttentionModule, self).get_config()
        base_config['in_channels'] = self.in_channels
        base_config['gamma_initializer'] = self.gamma_initializer
        base_config['gamma_regularizer'] = self.gamma_regularizer
        base_config['gamma_constraint'] = self.gamma_constraint
        return base_config


class DualAttentionModule(tf.keras.layers.Layer):
    def __init__(self, in_channels,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None):
        super(DualAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint
        
        self.cam = ChannelAttentionModule(gamma_initializer=gamma_initializer,
                                        gamma_regularizer=gamma_regularizer,
                                        gamma_constraint=gamma_constraint)
        self.pam = PositionAttentionModule(in_channels=in_channels, gamma_initializer=gamma_initializer,
                                        gamma_regularizer=gamma_regularizer,
                                        gamma_constraint=gamma_constraint)
    
    def call(self, x):

        feat_p = self.pam(x)
        feat_c = self.cam(x)
        feat_fusion = feat_p + feat_c

        return feat_fusion
    
    def get_config(self):
        base_config = super(DualAttentionModule, self).get_config()
        base_config['in_channels'] = self.in_channels
        base_config['gamma_initializer'] = self.gamma_initializer
        base_config['gamma_regularizer'] = self.gamma_regularizer
        base_config['gamma_constraint'] = self.gamma_constraint
        return base_config