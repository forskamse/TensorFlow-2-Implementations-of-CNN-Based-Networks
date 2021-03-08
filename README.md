# TensorFlow-2-Implementations-of-CNN-Based-Networks

The repository implements a list of CNN blocks/networks based on TensorFlow 2.x, catogorized into feature extraction/fusion blocks, attention blocks and backbone networks.

## [Codes]

### Feature Extraction/Fusion Blocks
- [Atrous Convolutional Block](./Feature_Extraction_and_Fusion_Blocks/Atrous_Convolutional_Block.py) for 1D (data points / sequences) or 2D inputs (images / feature maps), suggested by *[An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271)*

- [Receptive Field Block](./Feature_Extraction_and_Fusion_Blocks/Receptive_Field_Block.py), from *[Receptive Field Block Net for Accurate and Fast Object Detection](https://arxiv.org/pdf/1711.07767.pdf)*


### Attention Blocks
- [Squeeze-and-Excitation Block (Kind of Channel Attention)](./Attention_Blocks/Squeeze_and_Excitation_Block.py), from *[Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)*

- [Convolutional Block Attention Module (CBAM)](./Attention_Blocks/Convolutional_Block_Attention_Module.py), including Channel Attention Module and Spatial Attention Module, from *[CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)*

- [Non-Local Block](./Attention_Blocks/Non_Local_Block.py), including 'Gaussian', 'Embedded Gaussian', 'Dot Product' and 'Concatenation' modes, from *[Non-local Neural Networks](https://arxiv.org/pdf/1711.07971.pdf)*

- [Dual Attention Module](./Attention_Blocks/Dual_Attention_Module.py), including Channel Attention Module and Position Attention Module, from *[Dual Attention Network for Scene Segmentation](https://arxiv.org/pdf/1809.02983.pdf)*


### Backbone Networks
- [DenseNet](./Backbone_Networks/DenseNet.py), from *[Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)*


## [Demo]
The test data [test_data.npy](.test_data.npy) and [test.py](.test.py) script give a demo for using these CNN based blocks/networks.    
The test data include 100 samples with the shape (24, 16, 2).    
The demo implements a encoder-decoder model to conduct self-supervised learning for embeddings learning.    


## [Reference]

[1] https://github.com/philipperemy/keras-tcn    
[2] https://github.com/Baichenjia/Tensorflow-TCN/blob/master/tcn.py    
[3] https://arxiv.org/pdf/1803.01271.pdf    
[4] https://arxiv.org/pdf/1711.07767.pdf    
[5] https://arxiv.org/abs/1709.01507    
[6] https://github.com/kobiso/CBAM-tensorflow-slim/blob/master/nets/attention_module.py    
[7] https://arxiv.org/abs/1807.06521    
[8] https://arxiv.org/pdf/1711.07971.pdf    
[9] https://github.com/titu1994/keras-non-local-nets/blob/master/non_local.py    
[10] https://github.com/Tramac/Non-local-tensorflow/tree/master/non_local    
[11] https://arxiv.org/pdf/1809.02983.pdf    
[12] https://github.com/niecongchong/DANet-keras/blob/master/layers/attention.py    
[13] https://github.com/okason97/DenseNet-Tensorflow2/blob/master/densenet/densenet.py    
[14] https://arxiv.org/pdf/1608.06993.pdf    