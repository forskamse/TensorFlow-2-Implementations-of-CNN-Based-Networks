#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DenseNet.py
@Time    :   2021/03/05 22:43:46
@Author  :   Forskamse
@Version :   1.0
@Contact :   mb85453@@um.edu.mo
@License :   (C)Copyright 2019-2021, UM, SKL-IOTSC, MACAU, CHINA
@Desc    :   Tensorflow 2.x implementation of DenseNet from the paper: Densely Connected Convolutional Networks
@Ref     :   [1]https://github.com/okason97/DenseNet-Tensorflow2/blob/master/densenet/densenet.py [2]https://arxiv.org/pdf/1608.06993.pdf
'''

import tensorflow as tf
from tensorflow.keras import layers, Sequential

class dense_conv_block(tf.keras.layers.Layer):
    def __init__(self, in_channels, dropout_rate=None):
        
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate

        super(dense_conv_block, self).__init__()
        
        # 1x1 Convolution (Bottleneck layer)
        inter_channel = in_channels * 4
        self.bn1 = layers.BatchNormalization(axis=-1)
        self.ac1 = layers.Activation(activation='relu')
        self.conv1 = layers.Conv2D(inter_channel, kernel_size=1, strides=1)
        self.drop1 = layers.Dropout(dropout_rate)
        
        # 3x3 Convolution
        self.bn2 = layers.BatchNormalization(axis=-1)
        self.ac2 = layers.Activation(activation='relu')
        self.conv2 = layers.Conv2D(in_channels, kernel_size=3, strides=1, padding='same')
        self.drop2 = layers.Dropout(dropout_rate)

        
    def call(self, x, training):

        x = self.bn1(x)
        x = self.ac1(x)
        x = self.conv1(x)

        if training:
            x = self.drop1(x)
        
        x = self.bn2(x)
        x = self.ac2(x)
        x = self.conv2(x)

        if training:
            x = self.drop2(x)

        return x

class dense_block(tf.keras.layers.Layer):
    def __init__(self, num_layers, growth_rate, dropout_rate=None):
        
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.dropout_rate = dropout_rate

        super(dense_block, self).__init__()

        dense_conv_blocks = []
        for _ in range(self.num_layers):
            dense_conv_blocks.append(dense_conv_block(growth_rate, dropout_rate))
        
        self.dense_conv_blocks = dense_conv_blocks

    def call(self, x):
        
        concat_feat = x
        
        for i in range(self.num_layers):
            x = self.dense_conv_blocks[i](concat_feat)
            concat_feat = tf.concat([concat_feat, x], -1)

        return concat_feat

class se_block(tf.keras.layers.Layer):
    def __init__(self, in_channels, reduction_ratio = 16):
        super(se_block, self).__init__()

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

class transition_block(tf.keras.layers.Layer):
    def __init__(self, in_channels, compression=1.0, dropout_rate=None):

        super(transition_block, self).__init__()

        self.bn1 = layers.BatchNormalization(axis=-1)
        self.ac1 = layers.Activation(activation='relu')
        self.conv1 = layers.Conv2D(int(in_channels * compression), kernel_size=1, strides=1)
        self.drop1 = layers.Dropout(dropout_rate)
        self.avgpool1 = layers.AveragePooling2D((2, 2), strides=(2, 2))

    def call(self, x, training):
        
        x = self.bn1(x)
        x = self.ac1(x)
        x = self.conv1(x)

        if training:
            x = self.drop1(x)

        x = self.avgpool1(x)

        return x

class DenseNet(tf.keras.layers.Layer):
    def __init__(self, growth_rate=32, init_channels=64, num_layers = [6,12,24,16], reduction=0.0, 
                   dropout_rate=0.0, with_se_layers=True, **kwargs):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.init_channels = init_channels
        self.num_layers = num_layers
        self.reduction = reduction
        self.dropout_rate = dropout_rate
        self.with_se_layers = with_se_layers
    
        model = tf.keras.Sequential()
    
        in_channels = init_channels
        conv_first = layers.Conv2D(in_channels, kernel_size=3, strides=1, padding='same')
        bn_first = layers.BatchNormalization(axis=-1)
        ac_first = layers.Activation(activation='relu')
        
        model.add(conv_first)
        model.add(bn_first)
        model.add(ac_first)
        
        compression = 1.0 - reduction
        nb_dense_block = len(num_layers)

        for block_idx in range(nb_dense_block - 1):
            model.add(dense_block(num_layers[block_idx], growth_rate, dropout_rate))   
            in_channels = in_channels + growth_rate * num_layers[block_idx]

            if with_se_layers:
                model.add(se_block(in_channels))
            
            model.add(transition_block(in_channels, compression, dropout_rate))
            in_channels = int(in_channels * compression)

            if with_se_layers:
                model.add(se_block(in_channels))

        model.add(dense_block(num_layers[-1], growth_rate, dropout_rate))
        in_channels = in_channels + growth_rate * num_layers[-1]

        if with_se_layers:
            model.add(se_block(in_channels))
        
        
        bn_last = layers.BatchNormalization(axis=-1)
        ac_last = layers.Activation(activation='relu')
        global_avgpool = layers.GlobalAveragePooling2D()
    
        model.add(bn_last)
        model.add(ac_last)
        model.add(global_avgpool)

        self.network = model

    def call(self, x):
        return self.network(x)

    def get_config(self):
        
        base_config = super(DenseNet, self).get_config()
        base_config['growth_rate'] = self.growth_rate
        base_config['init_channels'] = self.init_channels
        base_config['num_layers'] = self.num_layers
        base_config['reduction'] = self.reduction
        base_config['dropout_rate'] = self.dropout_rate
        base_config['with_se_layers'] = self.with_se_layers
    
        return base_config