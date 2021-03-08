#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test.py
@Time    :   2021/03/08 12:34:38
@Author  :   Forskamse
@Version :   1.0
@Contact :   mb85453@@um.edu.mo
@License :   (C)Copyright 2019-2021, UM, SKL-IOTSC, MACAU, CHINA
@Desc    :   The script implements a encoder-decoder model to conduct self-supervised learning for embeddings learning.
             It is a demo for using the CNN based blocks/networks.
@Ref     :   None
'''

import os
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from Feature_Extraction_and_Fusion_Blocks import AtrousConvBlock_1D, AtrousConvBlock_2D, ReceptiveFieldBlock
from Attention_Blocks import SqueezeExcitationBlock, NonLocalBlock, CBAM, DualAttentionModule
from Backbone_Networks import DenseNet

def Encoder(enc_input, embed_dim):
    h = layers.Conv2D(32, kernel_size=(3, 3), padding='same')(enc_input)
    h = layers.BatchNormalization(axis=-1)(h)
    h = layers.Activation('relu')(h)
    # h = AtrousConvBlock_2D(dilation_rate=3, inter_channels=8, kernel_size=3, dropout_rate=0.5, padding='same')(h)
    # h = ReceptiveFieldBlock(inter_channels=8, out_channels=8)(h)
    # h = SqueezeExcitationBlock(in_channels=32, reduction_ratio = 16)(h)
    # h = NonLocalBlock(in_channels=32, inter_channels=16, out_channels=32, compression=2, mode='gaussian')(h)
    # h = NonLocalBlock(inter_channels=16, out_channels=32, compression=2, mode='dot_product')(h)
    # h = NonLocalBlock(inter_channels=16, out_channels=32, compression=2, mode='embedded_gaussian')(h)
    # h = NonLocalBlock(inter_channels=16, out_channels=32, compression=2, mode='concatenation')(h)
    # h = CBAM(in_channels=32, reduction_ratio=16)(h)
    # h = DualAttentionModule(in_channels=32)(h)
    h = DenseNet(growth_rate=32, init_channels=128, num_layers = [3,4,8,5], reduction=0.0, 
                   dropout_rate=0.0, with_se_layers=True)(h)
    h = layers.Reshape((24, 16, 2))(h) # Global Average Pooling in DenseNet Changes the shape
    h = layers.Conv2D(2, kernel_size=(3, 3), padding='same')(h)
    h = layers.BatchNormalization(axis=-1)(h)
    h = layers.Activation('relu')(h)
    h = layers.Conv2D(2, kernel_size=(3, 3), padding='same')(h)
    h = layers.BatchNormalization(axis=-1)(h)
    h = layers.Activation('relu')(h)
    h = layers.Flatten()(h)
    enc_output = layers.Dense(embed_dim)(h)
    return enc_output

def Decoder(dec_input, embed_dim):
    h = layers.Dense(768)(dec_input)
    h = layers.Reshape((24, 16, 2))(h)
    h = layers.Conv2D(2, kernel_size=(3, 3), padding='same')(h)
    h = layers.BatchNormalization(axis=-1)(h)
    h = layers.Activation('relu')(h)
    h = layers.Conv2D(2, kernel_size=(3, 3), padding='same')(h)
    h = layers.BatchNormalization(axis=-1)(h)
    h = layers.Activation('relu')(h)
    h = layers.Conv2D(2, kernel_size=(3, 3), padding='same')(h)
    h = layers.BatchNormalization(axis=-1)(h)
    h = layers.Activation('relu')(h)
    dec_output = layers.Dense(2, activation='sigmoid')(h)
    return dec_output

if __name__ == "__main__":
    
    embed_dim = 500

    #=======================================================================================================================
    #=======================================================================================================================
    # Data Loading
    SHAPE_DIM1 = 24
    SHAPE_DIM2 = 16
    SHAPE_DIM3 = 2
    data = np.load('test_data.npy')
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

    #=======================================================================================================================
    #=======================================================================================================================
    # Model Constructing
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # Encoder
        encInput = keras.Input(shape=(SHAPE_DIM1, SHAPE_DIM2, SHAPE_DIM3))
        encOutput = Encoder(encInput, embed_dim)
        encModel = keras.Model(inputs=encInput, outputs=encOutput, name='Encoder')
        # Decoder
        decInput = keras.Input(shape=(embed_dim,))
        decOutput = Decoder(decInput, embed_dim)
        decModel = keras.Model(inputs=decInput, outputs=decOutput, name="Decoder")
        # Autoencoder
        autoencoderInput = keras.Input(shape=(SHAPE_DIM1, SHAPE_DIM2, SHAPE_DIM3))
        autoencoderOutput = decModel(encModel(autoencoderInput))
        autoencoderModel = keras.Model(inputs=autoencoderInput, outputs=autoencoderOutput, name='Autoencoder')
        # Comliling
        autoencoderModel.compile(optimizer='adam', loss='mse')
    print(autoencoderModel.summary())
    print(encModel.summary())
    print(decModel.summary())

    autoencoderModel.fit(x=train_data, y=train_data, batch_size=32, epochs=200, validation_data = (test_data, test_data))