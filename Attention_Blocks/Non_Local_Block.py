#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Non_Local_Block.py
@Time    :   2021/03/06 20:37:08
@Author  :   Forskamse
@Version :   1.0
@Contact :   mb85453@@um.edu.mo
@License :   (C)Copyright 2019-2021, UM, SKL-IOTSC, MACAU, CHINA
@Desc    :   Tensorflow 2.x implementation of Non-Local Block, including 'Gaussian', 'Embedded Gaussian', 'Dot Product' and 'Concatenation' modes, from the paper Non-local Neural Networks.
@Ref     :   [1]https://arxiv.org/pdf/1711.07971.pdf [2]https://github.com/titu1994/keras-non-local-nets/blob/master/non_local.py [3]https://github.com/Tramac/Non-local-tensorflow/tree/master/non_local
'''

import tensorflow as tf
from tensorflow.keras import layers, Sequential

class NonLocalBlock(tf.keras.layers.Layer):
    
    def __init__(self, in_channels=None, inter_channels=64, out_channels=64, compression=None, mode='embedded_gaussian'):
        super(NonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.compression = compression
        self.mode = mode

        if mode not in ['gaussian', 'embedded_gaussian', 'dot_product', 'concatenation']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded_gaussian`, `dot_product` or `concatenation`')

        if mode == 'gaussian':
            self.re1 = layers.Reshape((-1, in_channels))
            self.re2 = layers.Reshape((-1, in_channels))
            self.ac = layers.Activation('softmax')
            if compression:
                self.maxpool_phi = layers.MaxPool1D(compression)
                self.maxpool_g = layers.MaxPool1D(compression)
                
        elif mode == 'dot_product':
            self.conv1 = layers.Conv2D(inter_channels, kernel_size=1, padding='same')
            self.re1 = layers.Reshape((-1, inter_channels))
            self.conv2 = layers.Conv2D(inter_channels, kernel_size=1, padding='same')
            self.re2 = layers.Reshape((-1, inter_channels))
            if compression:
                self.maxpool_phi = layers.MaxPool1D(compression)
                self.maxpool_g = layers.MaxPool1D(compression)

        elif mode == 'embedded_gaussian':
            self.conv1 = layers.Conv2D(inter_channels, kernel_size=1, padding='same')
            self.re1 = layers.Reshape((-1, inter_channels))
            self.conv2 = layers.Conv2D(inter_channels, kernel_size=1, padding='same')
            self.re2 = layers.Reshape((-1, inter_channels))
            self.ac = layers.Activation('softmax')
            if compression:
                self.maxpool_phi = layers.MaxPool1D(compression)
                self.maxpool_g = layers.MaxPool1D(compression)

        elif mode == 'concatenation':
            self.conv1 = layers.Conv2D(inter_channels, kernel_size=1, padding='same')
            self.re1 = layers.Reshape((-1, 1, inter_channels))
            self.conv2 = layers.Conv2D(inter_channels, kernel_size=1, padding='same')
            self.re2 = layers.Reshape((1, -1, inter_channels))
            self.concat_project = layers.Conv2D(1, kernel_size=1, padding='same')
            if compression:
                self.maxpool_phi = layers.MaxPool2D(compression)
                self.maxpool_g = layers.MaxPool2D(compression)

        self.conv_g = layers.Conv2D(inter_channels, kernel_size=1, padding='same')
        self.re_g = layers.Reshape((-1, inter_channels))

        self.conv_project = layers.Conv2D(out_channels, kernel_size=1, padding='same')

    def call(self, x):

        prev_x = x

        _w, _h = tf.shape(x)[1], tf.shape(x)[2]
        
        if self.mode == 'gaussian':
            theta = self.re1(x)
            phi = self.re2(x)
            if self.compression:
                phi = self.maxpool_phi(phi)
            f = layers.dot([theta, phi], axes=2)
            f = self.ac(f)

            g = self.conv_g(x)
            g = self.re_g(g)
            if self.compression:
                g = self.maxpool_g(g)

        elif self.mode == 'dot_product':
            theta = self.conv1(x)
            theta = self.re1(theta)
            phi = self.conv2(x)
            phi = self.re2(phi)
            if self.compression:
                phi = self.maxpool_phi(phi)
            f = layers.dot([theta, phi], axes=2)
            f = f / tf.cast(tf.shape(f)[-1], tf.float32)

            g = self.conv_g(x)
            g = self.re_g(g)
            if self.compression:
                g = self.maxpool_g(g)
        
        elif self.mode == 'embedded_gaussian':
            theta = self.conv1(x)
            theta = self.re1(theta)
            phi = self.conv2(x)
            phi = self.re2(phi)
            if self.compression:
                phi = self.maxpool_phi(phi)
            f = layers.dot([theta, phi], axes=2)
            f = self.ac(f)

            g = self.conv_g(x)
            g = self.re_g(g)
            if self.compression:
                g = self.maxpool_g(g)

        elif self.mode == 'concatenation':
            theta = self.conv1(x)
            theta = self.re1(theta)
            phi = self.conv2(x)
            if self.compression:
                phi = self.maxpool_phi(phi)
            phi = self.re2(phi)
            h = tf.shape(theta)[1]
            w = tf.shape(phi)[2]
            theta = tf.tile(theta, (1, 1, w, 1))
            phi = tf.tile(phi, (1, h, 1, 1))
            f = layers.Concatenate(axis=-1)([theta, phi])
            f = self.concat_project(f)
            f = layers.Reshape((h,w))(f)
            print(f.shape)
            f = f / tf.cast(tf.shape(f)[-1], tf.float32)
            
            
            g = self.conv_g(x)
            if self.compression:
                g = self.maxpool_g(g)
            g = self.re_g(g)

        x = layers.dot([f, g], axes=[2, 1])
        x = layers.Reshape((_w, _h, self.inter_channels))(x)
        x = self.conv_project(x)
        x = layers.add([prev_x, x])

        return x

    def get_config(self):
        base_config = super(NonLocalBlock, self).get_config()
        base_config['in_channels'] = self.in_channels
        base_config['inter_channels'] = self.inter_channels
        base_config['out_channels'] = self.out_channels
        base_config['mode'] = self.mode
        base_config['compression'] = self.compression
        return base_config