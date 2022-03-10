# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/9 16:33
# @Author  : dongchao yang
# @File    : train.py
from itertools import zip_longest
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import math
from sklearn.cluster import KMeans
import os
import time
from functools import partial
# import timm
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import warnings
from functools import partial
# from timm.models.registry import register_model
# from timm.models.vision_transformer import _cfg
# from mmdet.utils import get_root_logger
# from mmcv.runner import load_checkpoint
# from mmcv.runner import _load_checkpoint, load_state_dict
# import mmcv.runner
import copy
from collections import OrderedDict
import io
import re
DEBUG=0

def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None,
                    revise_keys=[(r'^module\.', '')]):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Default: strip
            the prefix 'module.' by [(r'^module\\.', '')].

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    checkpoint = _load_checkpoint(filename, map_location, logger)
    '''
    new_proj = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    new_proj.weight = torch.nn.Parameter(torch.sum(checkpoint['patch_embed1.proj.weight'], dim=1).unsqueeze(1))
    checkpoint['patch_embed1.proj.weight'] = new_proj.weight
    new_proj.weight = torch.nn.Parameter(torch.sum(checkpoint['patch_embed1.proj.weight'], dim=2).unsqueeze(2).repeat(1,1,3,1))
    checkpoint['patch_embed1.proj.weight'] = new_proj.weight
    new_proj.weight = torch.nn.Parameter(torch.sum(checkpoint['patch_embed1.proj.weight'], dim=3).unsqueeze(3).repeat(1,1,1,3))
    checkpoint['patch_embed1.proj.weight'] = new_proj.weight
    '''
    new_proj = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
    new_proj.weight = torch.nn.Parameter(torch.sum(checkpoint['patch_embed1.proj.weight'], dim=1).unsqueeze(1))
    checkpoint['patch_embed1.proj.weight'] = new_proj.weight
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in state_dict.items()})
    state_dict = OrderedDict({k.replace('backbone.',''):v for k,v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class MaxPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.max(decision, dim=self.pooldim)[0]


class LinearSoftPool(nn.Module):
    """LinearSoftPool
    Linear softmax, takes logits and returns a probability, near to the actual maximum value.
    Taken from the paper:
        A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling
    https://arxiv.org/abs/1810.09050

    """
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, time_decision):
        return (time_decision**2).sum(self.pooldim) / (time_decision.sum(
            self.pooldim)+1e-7)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class ConvBlock_GLU(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=(3,3)):
        super(ConvBlock_GLU, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size, stride=(1, 1),
                              padding=(1, 1), bias=False)                         
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = self.bn1(self.conv1(x))
        cnn1 = self.sigmoid(x[:, :x.shape[1]//2, :, :])
        cnn2 = x[:,x.shape[1]//2:,:,:]
        x = cnn1*cnn2
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        elif pool_type == 'None':
            pass
        elif pool_type == 'LP':
            pass
            #nn.LPPool2d(4, pool_size)
        else:
            raise Exception('Incorrect argument!')
        return x

class Mul_scale_GLU(nn.Module):
    def __init__(self):
        super(Mul_scale_GLU,self).__init__()
        self.conv_block1_1 = ConvBlock_GLU(in_channels=1, out_channels=64,kernel_size=(1,1)) # 1*1
        self.conv_block1_2 = ConvBlock_GLU(in_channels=1, out_channels=64,kernel_size=(3,3)) # 3*3
        self.conv_block1_3 = ConvBlock_GLU(in_channels=1, out_channels=64,kernel_size=(5,5)) # 5*5
        self.conv_block2 = ConvBlock_GLU(in_channels=96, out_channels=128*2)
        # self.conv_block3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock_GLU(in_channels=128, out_channels=128*2)
        self.conv_block4 = ConvBlock_GLU(in_channels=128, out_channels=256*2)
        self.conv_block5 = ConvBlock_GLU(in_channels=256, out_channels=256*2)
        self.conv_block6 = ConvBlock_GLU(in_channels=256, out_channels=512*2)
        self.conv_block7 = ConvBlock_GLU(in_channels=512, out_channels=512*2)
        self.padding = nn.ReplicationPad2d((0,1,0,1))

    def forward(self, input, fi=None):
        """
        Input: (batch_size, data_length)"""
        x1 = self.conv_block1_1(input, pool_size=(2, 2), pool_type='avg')
        x1 = x1[:,:,:500,:32]
        #print('x1 ',x1.shape)
        x2 = self.conv_block1_2(input,pool_size=(2,2),pool_type='avg')
        #print('x2 ',x2.shape)
        x3 = self.conv_block1_3(input,pool_size=(2,2),pool_type='avg')
        x3 = self.padding(x3)
        #print('x3 ',x3.shape)
        # assert 1==2
        x = torch.cat([x1,x2],dim=1)
        x = torch.cat([x,x3],dim=1)
        #print('x ',x.shape)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='None')
        x = self.conv_block3(x,pool_size=(2,2),pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training) # 
        #print('x2,3 ',x.shape)
        x = self.conv_block4(x, pool_size=(2, 4), pool_type='None')
        x = self.conv_block5(x,pool_size=(2,4),pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        #print('x4,5 ',x.shape)

        x = self.conv_block6(x, pool_size=(1, 4), pool_type='None')
        x = self.conv_block7(x, pool_size=(1, 4), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print('x6,7 ',x.shape)
        # assert 1==2
        return x

class Cnn14(nn.Module):
    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, 
        fmax=14000, classes_num=527):
        
        super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 128, bias=True)
        self.fc_audioset = nn.Linear(128, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input_, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        input_ = input_.unsqueeze(1)
        x = self.conv_block1(input_, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x.shape)
        # x = torch.mean(x, dim=3)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x = self.fc1(x)
        # print(x.shape)
        # assert 1==2
        # (x1,_) = torch.max(x, dim=2)
        # x2 = torch.mean(x, dim=2)
        # x = x1 + x2
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = F.relu_(self.fc1(x))
        # embedding = F.dropout(x, p=0.5, training=self.training)
        return x

class Cnn10_fi(nn.Module):
    def __init__(self):  
        super(Cnn10_fi, self).__init__()
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        # self.fc1 = nn.Linear(512, 512, bias=True)
        # self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        
        # self.init_weight()
 
    def forward(self, input, fi=None):
        """
        Input: (batch_size, data_length)"""

        x = self.conv_block1(input, pool_size=(2, 2), pool_type='avg')
        if fi != None:
            gamma = fi[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            beta = fi[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            x = (gamma)*x + beta
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        if fi != None:
            gamma = fi[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            beta = fi[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            x = (gamma)*x + beta
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 4), pool_type='avg')
        if fi != None:
            gamma = fi[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            beta = fi[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            x = (gamma)*x + beta
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 4), pool_type='avg')
        if fi != None:
            gamma = fi[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            beta = fi[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            x = (gamma)*x + beta
        x = F.dropout(x, p=0.2, training=self.training)
        return x

class Cnn10_mul_scale(nn.Module):
    def __init__(self,scale=8):  
        super(Cnn10_mul_scale, self).__init__()
        self.conv_block1_1 = ConvBlock_GLU(in_channels=1, out_channels=64,kernel_size=(1,1))
        self.conv_block1_2 = ConvBlock_GLU(in_channels=1, out_channels=64,kernel_size=(3,3))
        self.conv_block1_3 = ConvBlock_GLU(in_channels=1, out_channels=64,kernel_size=(5,5))
        self.conv_block2 = ConvBlock(in_channels=96, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.scale = scale
        self.padding = nn.ReplicationPad2d((0,1,0,1))
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        """
        Input: (batch_size, data_length)"""
        if self.scale == 8:
            pool_size1 = (2,2)
            pool_size2 = (2,2)
            pool_size3 = (2,4)
            pool_size4 = (1,4)
        elif self.scale == 4:
            pool_size1 = (2,2)
            pool_size2 = (2,2)
            pool_size3 = (1,4)
            pool_size4 = (1,4)
        elif self.scale == 2:
            pool_size1 = (2,2)
            pool_size2 = (1,2)
            pool_size3 = (1,4)
            pool_size4 = (1,4)
        else:
            pool_size1 = (1,2)
            pool_size2 = (1,2)
            pool_size3 = (1,4)
            pool_size4 = (1,4)
        # print('input ',input.shape)
        x1 = self.conv_block1_1(input, pool_size=pool_size1, pool_type='avg')
        x1 = x1[:,:,:500,:32]
        # print('x1 ',x1.shape)
        x2 = self.conv_block1_2(input, pool_size=pool_size1, pool_type='avg')
        # print('x2 ',x2.shape)
        x3 = self.conv_block1_3(input, pool_size=pool_size1, pool_type='avg')
        x3 = self.padding(x3)
        # print('x3 ',x3.shape)
        # assert 1==2
        x = torch.cat([x1,x2,x3],dim=1)
        # x = torch.cat([x,x3],dim=1)

        # x = self.conv_block1(input, pool_size=pool_size1, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=pool_size2, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=pool_size3, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=pool_size4, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        return x


class Cnn10(nn.Module):
    def __init__(self,scale=8):  
        super(Cnn10, self).__init__()
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.scale = scale
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        """
        Input: (batch_size, data_length)"""
        if self.scale == 8:
            pool_size1 = (2,2)
            pool_size2 = (2,2)
            pool_size3 = (2,4)
            pool_size4 = (1,4)
        elif self.scale == 4:
            pool_size1 = (2,2)
            pool_size2 = (2,2)
            pool_size3 = (1,4)
            pool_size4 = (1,4)
        elif self.scale == 2:
            pool_size1 = (2,2)
            pool_size2 = (1,2)
            pool_size3 = (1,4)
            pool_size4 = (1,4)
        else:
            pool_size1 = (1,2)
            pool_size2 = (1,2)
            pool_size3 = (1,4)
            pool_size4 = (1,4)
        x = self.conv_block1(input, pool_size=pool_size1, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=pool_size2, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=pool_size3, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=pool_size4, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        return x

class MeanPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.mean(decision, dim=self.pooldim)

class ResPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim
        self.linPool = LinearSoftPool(pooldim=1)

class AutoExpPool(nn.Module):
    def __init__(self, outputdim=10, pooldim=1):
        super().__init__()
        self.outputdim = outputdim
        self.alpha = nn.Parameter(torch.full((outputdim, ), 1))
        self.pooldim = pooldim

    def forward(self, logits, decision):
        scaled = self.alpha * decision  # \alpha * P(Y|x) in the paper
        return (logits * torch.exp(scaled)).sum(
            self.pooldim) / torch.exp(scaled).sum(self.pooldim)


class SoftPool(nn.Module):
    def __init__(self, T=1, pooldim=1):
        super().__init__()
        self.pooldim = pooldim
        self.T = T

    def forward(self, logits, decision):
        w = torch.softmax(decision / self.T, dim=self.pooldim)
        return torch.sum(decision * w, dim=self.pooldim)


class AutoPool(nn.Module):
    """docstring for AutoPool"""
    def __init__(self, outputdim=10, pooldim=1):
        super().__init__()
        self.outputdim = outputdim
        self.alpha = nn.Parameter(torch.ones(outputdim))
        self.dim = pooldim

    def forward(self, logits, decision):
        scaled = self.alpha * decision  # \alpha * P(Y|x) in the paper
        weight = torch.softmax(scaled, dim=self.dim)
        return torch.sum(decision * weight, dim=self.dim)  # B x C


class ExtAttentionPool(nn.Module):
    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):
        super().__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.pooldim = pooldim
        self.attention = nn.Linear(inputdim, outputdim)
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)
        self.activ = nn.Softmax(dim=self.pooldim)

    def forward(self, logits, decision):
        # Logits of shape (B, T, D), decision of shape (B, T, C)
        w_x = self.activ(self.attention(logits) / self.outputdim)
        h = (logits.permute(0, 2, 1).contiguous().unsqueeze(-2) *
             w_x.unsqueeze(-1)).flatten(-2).contiguous()
        return torch.sum(h, self.pooldim)


class AttentionPool(nn.Module):
    """docstring for AttentionPool"""
    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):
        super().__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.pooldim = pooldim
        self.transform = nn.Linear(inputdim, outputdim)
        self.activ = nn.Softmax(dim=self.pooldim)
        self.eps = 1e-7

    def forward(self, logits, decision):
        # Input is (B, T, D)
        # B, T , D
        w = self.activ(torch.clamp(self.transform(logits), -15, 15))
        detect = (decision * w).sum(
            self.pooldim) / (w.sum(self.pooldim) + self.eps)
        # B, T, D
        return detect

class Block2D(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, x):
        return self.block(x)

class AudioCNN(nn.Module):
    def __init__(self, classes_num):
        super(AudioCNN, self).__init__()
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc1 = nn.Linear(512,128,bias=True)
        self.fc = nn.Linear(128, classes_num, bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        # [128, 801, 168] --> [128,1,801,168]
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg') # 128,64,400,84
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg') # 128,128,200,42
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg') # 128,256,100,21
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg') # 128,512,50,10
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes) # 128,512,50
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps) 128,512
        x = self.fc1(x) # 128,128
        output = self.fc(x) # 128,10
        return x,output

    def extract(self,input):
        '''Input: (batch_size, times_steps, freq_bins)'''
        x = input[:, None, :, :]
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = self.fc1(x) # 128,128
        return x

def parse_poolingfunction(poolingfunction_name='mean', **kwargs):
    """parse_poolingfunction
    A heler function to parse any temporal pooling
    Pooling is done on dimension 1

    :param poolingfunction_name:
    :param **kwargs:
    """
    poolingfunction_name = poolingfunction_name.lower()
    if poolingfunction_name == 'mean':
        return MeanPool(pooldim=1)
    elif poolingfunction_name == 'max':
        return MaxPool(pooldim=1)
    elif poolingfunction_name == 'linear':
        return LinearSoftPool(pooldim=1)
    elif poolingfunction_name == 'expalpha':
        return AutoExpPool(outputdim=kwargs['outputdim'], pooldim=1)

    elif poolingfunction_name == 'soft':
        return SoftPool(pooldim=1)
    elif poolingfunction_name == 'auto':
        return AutoPool(outputdim=kwargs['outputdim'])
    elif poolingfunction_name == 'attention':
        return AttentionPool(inputdim=kwargs['inputdim'],
                             outputdim=kwargs['outputdim'])
class conv1d(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding='VALID', dilation=1):
        super(conv1d, self).__init__()
        if padding == 'VALID':
            dconv_pad = 0
        elif padding == 'SAME':
            dconv_pad = dilation * ((kernel_size - 1) // 2)
        else:
            raise ValueError("Padding Mode Error!")
        self.conv = nn.Conv1d(nin, nout, kernel_size=kernel_size, stride=stride, padding=dconv_pad)
        self.act = nn.ReLU()
        self.init_layer(self.conv)

    def init_layer(self, layer, nonlinearity='relu'):
        """Initialize a Linear or Convolutional layer. """
        nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
        nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        out = self.act(self.conv(x))
        return out

class Atten_1(nn.Module):
    def __init__(self, input_dim, context=2, dropout_rate=0.2):
        super(Atten_1, self).__init__()
        self._matrix_k = nn.Linear(input_dim, input_dim // 4)
        self._matrix_q = nn.Linear(input_dim, input_dim // 4)
        self.relu = nn.ReLU()
        self.context = context
        self._dropout_layer = nn.Dropout(dropout_rate)
        self.init_layer(self._matrix_k)
        self.init_layer(self._matrix_q)

    def init_layer(self, layer, nonlinearity='leaky_relu'):
        """Initialize a Linear or Convolutional layer. """
        nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

    def forward(self, input_x):
        k_x = input_x
        k_x = self.relu(self._matrix_k(k_x))
        k_x = self._dropout_layer(k_x)
        # print('k_x ',k_x.shape)
        q_x = input_x[:, self.context, :]
        # print('q_x ',q_x.shape)
        q_x = q_x[:, None, :]
        # print('q_x1 ',q_x.shape)
        q_x = self.relu(self._matrix_q(q_x))
        q_x = self._dropout_layer(q_x)
        # print('q_x2 ',q_x.shape)
        x_ = torch.matmul(k_x, q_x.transpose(-2, -1) / math.sqrt(k_x.size(-1)))
        # print('x_ ',x_.shape)
        x_ = x_.squeeze(2)
        alpha = F.softmax(x_, dim=-1)
        att_ = alpha
        # print('alpha ',alpha)
        alpha = alpha.unsqueeze(2).repeat(1,1,input_x.shape[2])
        # print('alpha ',alpha)
        # alpha = alpha.view(alpha.size(0), alpha.size(1), alpha.size(2), 1)
        out = alpha * input_x
        # print('out ', out.shape)
        # out = out.mean(2)
        out = out.mean(1)
        # print('out ',out.shape)
        # assert 1==2
        #y = alpha * input_x
        #return y, att_
        out = input_x[:, self.context, :] + out
        return out

class Fusion(nn.Module):
    def __init__(self, inputdim, inputdim2, n_fac):
        super().__init__()
        self.fuse_layer1 = conv1d(inputdim, inputdim2*n_fac,1)
        self.fuse_layer2 = conv1d(inputdim2, inputdim2*n_fac,1)
        self.avg_pool = nn.AvgPool1d(n_fac, stride=n_fac) # 沿着最后一个维度进行pooling

    def forward(self,embedding,mix_embed):
        embedding = embedding.permute(0,2,1)
        fuse1_out = self.fuse_layer1(embedding) # [2, 501, 2560] ,512*5, 1D卷积融合,spk_embeding ,扩大其维度 
        fuse1_out = fuse1_out.permute(0,2,1)

        mix_embed = mix_embed.permute(0,2,1)
        fuse2_out = self.fuse_layer2(mix_embed) # [2, 501, 2560] ,512*5, 1D卷积融合,spk_embeding ,扩大其维度 
        fuse2_out = fuse2_out.permute(0,2,1)
        as_embs = torch.mul(fuse1_out, fuse2_out) # 相乘 [2, 501, 2560]
        # (10, 501, 512)
        as_embs = self.avg_pool(as_embs) # [2, 501, 512] 相当于 2560//5
        return as_embs

class CDur_fusion(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(128, 128, bidirectional=True, batch_first=True)
        self.fusion = Fusion(128,2)
        self.fc = nn.Linear(256,256)
        self.outputlayer = nn.Linear(256, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, embedding): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = self.fusion(embedding,x)
        #x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class CDur(nn.Module):
    def __init__(self, inputdim, outputdim,time_resolution, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(256, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512,256)
        self.outputlayer = nn.Linear(256, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, embedding,one_hot=None): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class CDur_big(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            Block2D(1, 64),
            Block2D(64, 64),
            nn.LPPool2d(4, (2, 2)),
            Block2D(64, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 2)),
            Block2D(128, 256),
            Block2D(256, 256),
            nn.LPPool2d(4, (2, 4)),
            Block2D(256, 512),
            Block2D(512, 512),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),)
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(640, 512, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, embedding): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class CDur_GLU(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = Mul_scale_GLU()
        # with torch.no_grad():
        #     rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
        #     rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(640, 512,1, bidirectional=True, batch_first=True) # previous is 640
        # self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, outputdim)
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
        
    def forward(self, x, embedding,one_hot=None): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)

        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class CDur_CNN14(nn.Module):
    def __init__(self, inputdim, outputdim,time_resolution,**kwargs):
        super().__init__()
        if time_resolution==125:
            self.features = Cnn10(8)
        elif time_resolution == 250:
            #print('time_resolution ',time_resolution)
            self.features = Cnn10(4)
        elif time_resolution == 500:
            self.features = Cnn10(2)
        else:
            self.features = Cnn10(0)
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        # self.features = Cnn10()
        self.gru = nn.GRU(640, 512, bidirectional=True, batch_first=True)
        # self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, outputdim)
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
    
    def forward(self, x, embedding,one_hot=None):
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class CDur_CNN_mul_scale(nn.Module):
    def __init__(self, inputdim, outputdim,time_resolution,**kwargs):
        super().__init__()
        if time_resolution==125:
            self.features = Cnn10_mul_scale(8)
        elif time_resolution == 250:
            #print('time_resolution ',time_resolution)
            self.features = Cnn10_mul_scale(4)
        elif time_resolution == 500:
            self.features = Cnn10_mul_scale(2)
        else:
            self.features = Cnn10_mul_scale(0)
        # with torch.no_grad():
        #     rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
        #     rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        # self.features = Cnn10()
        self.gru = nn.GRU(640, 512, bidirectional=True, batch_first=True)
        # self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, outputdim)
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
    
    def forward(self, x, embedding,one_hot=None):
        # print('x ',x.shape)
        # assert 1==2
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class CDur_CNN_mul_scale_fusion(nn.Module):
    def __init__(self, inputdim, outputdim, time_resolution,**kwargs):
        super().__init__()
        if time_resolution==125:
            self.features = Cnn10_mul_scale(8)
        elif time_resolution == 250:
            #print('time_resolution ',time_resolution)
            self.features = Cnn10_mul_scale(4)
        elif time_resolution == 500:
            self.features = Cnn10_mul_scale(2)
        else:
            self.features = Cnn10_mul_scale(0)
        # with torch.no_grad():
        #     rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
        #     rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        # self.features = Cnn10()
        self.gru = nn.GRU(512, 512, bidirectional=True, batch_first=True)
        # self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.fusion = Fusion(128,512,2)
        self.outputlayer = nn.Linear(256, outputdim)
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
    
    def forward(self, x, embedding,one_hot=None):
        # print('x ',x.shape)
        # assert 1==2
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = self.fusion(embedding, x)
        #x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class RaDur_fusion_fix(nn.Module):
    def __init__(self, model_config, inputdim, outputdim, time_resolution, **kwargs):
        super().__init__()
        self.encoder = Cnn14()
        self.detection = CDur_CNN_mul_scale_fusion(inputdim, outputdim, time_resolution)
        self.softmax = nn.Softmax(dim=2)
        #self.temperature = 5
        if model_config['pre_train']:
            self.encoder.load_state_dict(torch.load(model_config['encoder_path'])['model'])
            self.detection.load_state_dict(torch.load(model_config['CDur_path']))
        for p in self.encoder.parameters(): # fix the parameter of s_student
            p.requires_grad = False
        self.q = nn.Linear(128,128)
        self.k = nn.Linear(128,128)
        self.q_ee = nn.Linear(128, 128)
        self.k_ee = nn.Linear(128, 128)
        self.temperature = 11.3 # sqrt(128)
        self.att_pool = model_config['att_pool']
        self.enhancement = model_config['enhancement'] 
        self.tao = model_config['tao']
        self.top = model_config['top']
        self.bn = nn.BatchNorm1d(128)
        self.EE_fusion = Fusion(128, 128, 4)

    def get_w(self,q,k):
        q = self.q(q)
        k = self.k(k)
        q = q.unsqueeze(1)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn/self.temperature
        attn = self.softmax(attn)
        return attn
    
    def get_w_ee(self,q,k):
        q = self.q_ee(q)
        k = self.k_ee(k)
        q = q.unsqueeze(1)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn/self.temperature
        attn = self.softmax(attn)
        return attn
    
    def attention_pooling(self, embeddings, mean_embedding):
        att_pool_w = self.get_w(mean_embedding,embeddings)
        embedding = torch.bmm(att_pool_w, embeddings).squeeze(1)
        # print(embedding.shape)
        # print(att_pool_w.shape)
        # print(att_pool_w[0])
        # assert 1==2
        return embedding
    
    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1) # 根据分数进行排序
        top_k = _[:,:k]
        # print('top_k ', top_k)
        # top_k = top_k.mean(1)
        idx_topk = idx_DESC[:, :k] # 取top_k个
        # print('index ', idx_topk)
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings,top_k
    
    def sum_with_attention(self, embedding, top_k, selected_embeddings):
        # print('embedding ',embedding)
        # print('selected_embeddings ',selected_embeddings.shape)
        att_1 = self.get_w_ee(embedding, selected_embeddings)
        att_1 = att_1.squeeze(1)
        #print('att_1 ',att_1.shape)
        larger = top_k > self.tao
        # print('larger ',larger)
        top_k = top_k*larger
        # print('top_k ',top_k.shape)
        # print('top_k ',top_k)
        att_1 = att_1*top_k
        #print('att_1 ',att_1.shape)
        # assert 1==2
        att_2 = att_1.unsqueeze(2).repeat(1,1,128)
        Es = selected_embeddings*att_2
        return Es
    
    def orcal_EE(self, x, embedding, label):
        batch, time, dim = x.shape

        mixture_embedding = self.encoder(x) # 8, 125, 128
        mixture_embedding = mixture_embedding.transpose(1,2)
        mixture_embedding = self.bn(mixture_embedding)
        mixture_embedding = mixture_embedding.transpose(1,2)

        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding_pre = embedding.unsqueeze(1)
        embedding_pre = embedding_pre.repeat(1, x.shape[1], 1)
        f = self.detection.fusion(embedding_pre, x) # the first stage results
        #f = torch.cat((x, embedding_pre), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        f, _ = self.detection.gru(f) #  x  torch.Size([16, 125, 256])
        f = self.detection.fc(f)
        decision_time = torch.softmax(self.detection.outputlayer(f),dim=2) # x  torch.Size([16, 125, 2])
        
        selected_embeddings, top_k = self.select_topk_embeddings(decision_time[:,:,0], mixture_embedding, self.top)
        #selected_embeddings, top_k  = self.select_topk_embeddings(label, mixture_embedding, self.top)
        
        selected_embeddings = self.sum_with_attention(embedding, top_k, selected_embeddings) # add the weight

        mix_embedding = selected_embeddings.mean(1).unsqueeze(1) # 
        mix_embedding = mix_embedding.repeat(1, x.shape[1], 1)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        mix_embedding = self.EE_fusion(mix_embedding, embedding) # 使用神经网络进行融合
        # mix_embedding2 = selected_embeddings2.mean(1)
        #mix_embedding =  embedding + mix_embedding # 直接相加
        # new detection results
        # embedding_now = mix_embedding.unsqueeze(1)
        # embedding_now = embedding_now.repeat(1, x.shape[1], 1)
        f_now = self.detection.fusion(mix_embedding, x) 
        #f_now = torch.cat((x, embedding_now), dim=2) # 
        f_now, _ = self.detection.gru(f_now) #  x  torch.Size([16, 125, 256])
        f_now = self.detection.fc(f_now)
        decision_time_now = torch.softmax(self.detection.outputlayer(f_now), dim=2) # x  torch.Size([16, 125, 2])
        
        top_k = top_k.mean(1)  # get avg score,higher score will have more weight
        larger = top_k > self.tao
        top_k = top_k * larger
        top_k = top_k/2.0
        # print('top_k ',top_k)
        # assert 1==2
        # print('tok_k[ ',top_k.shape)
        # print('decision_time ',decision_time.shape)
        # print('decision_time_now ',decision_time_now.shape)
        neg_w = top_k.unsqueeze(1).unsqueeze(2)
        neg_w = neg_w.repeat(1, decision_time_now.shape[1], decision_time_now.shape[2])
        # print('neg_w ',neg_w.shape)
        #print('neg_w ',neg_w[:,0:10,0])
        pos_w = 1-neg_w
        #print('pos_w ',pos_w[:,0:10,0])
        decision_time_final = decision_time*pos_w + neg_w*decision_time_now
        #print('decision_time_final ',decision_time_final[0,0:10,0])
        # print(decision_time_final[0,:,:])
        #assert 1==2
        return decision_time_final
    
    def forward(self, x, ref, label=None):
        batch, time, dim = x.shape
        logit = torch.zeros(1).cuda()
        embeddings  = self.encoder(ref)
        mean_embedding = embeddings.mean(1)
        if self.att_pool == True:
            mean_embedding = self.bn(mean_embedding)
            embeddings = embeddings.transpose(1,2)
            embeddings = self.bn(embeddings)
            embeddings = embeddings.transpose(1,2)
            embedding = self.attention_pooling(embeddings, mean_embedding)
        else:
            embedding = mean_embedding
        if self.enhancement == True:
            decision_time = self.orcal_EE(x, embedding, label)
            decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
            return decision_time[:,:,0], decision_up, logit

        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        # x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        x = self.detection.fusion(embedding, x) 
        # embedding = embedding.unsqueeze(1)
        # embedding = embedding.repeat(1, x.shape[1], 1)
        # x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        x, _ = self.detection.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.detection.fc(x)
        decision_time = torch.softmax(self.detection.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2),
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0], decision_up, logit


class RaDur_fusion(nn.Module):
    def __init__(self, model_config, inputdim, outputdim, time_resolution, **kwargs):
        super().__init__()
        self.encoder = Cnn14()
        self.detection = CDur_CNN_mul_scale_fusion(inputdim, outputdim, time_resolution)
        self.softmax = nn.Softmax(dim=2)
        #self.temperature = 5
        if model_config['pre_train']:
            self.encoder.load_state_dict(torch.load(model_config['encoder_path'])['model'])
            self.detection.load_state_dict(torch.load(model_config['CDur_path']))
        
        self.q = nn.Linear(128,128)
        self.k = nn.Linear(128,128)
        self.q_ee = nn.Linear(128, 128)
        self.k_ee = nn.Linear(128, 128)
        self.temperature = 11.3 # sqrt(128)
        self.att_pool = model_config['att_pool']
        self.enhancement = model_config['enhancement'] 
        self.tao = model_config['tao']
        self.top = model_config['top']
        self.bn = nn.BatchNorm1d(128)
        self.EE_fusion = Fusion(128, 128, 4)

    def get_w(self,q,k):
        q = self.q(q)
        k = self.k(k)
        q = q.unsqueeze(1)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn/self.temperature
        attn = self.softmax(attn)
        return attn
    
    def get_w_ee(self,q,k):
        q = self.q_ee(q)
        k = self.k_ee(k)
        q = q.unsqueeze(1)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn/self.temperature
        attn = self.softmax(attn)
        return attn
    
    def attention_pooling(self, embeddings, mean_embedding):
        att_pool_w = self.get_w(mean_embedding,embeddings)
        embedding = torch.bmm(att_pool_w, embeddings).squeeze(1)
        # print(embedding.shape)
        # print(att_pool_w.shape)
        # print(att_pool_w[0])
        # assert 1==2
        return embedding
    
    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1) # 根据分数进行排序
        top_k = _[:,:k]
        # print('top_k ', top_k)
        # top_k = top_k.mean(1)
        idx_topk = idx_DESC[:, :k] # 取top_k个
        # print('index ', idx_topk)
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings,top_k
    
    def sum_with_attention(self, embedding, top_k, selected_embeddings):
        # print('embedding ',embedding)
        # print('selected_embeddings ',selected_embeddings.shape)
        att_1 = self.get_w_ee(embedding, selected_embeddings)
        att_1 = att_1.squeeze(1)
        #print('att_1 ',att_1.shape)
        larger = top_k > self.tao
        # print('larger ',larger)
        top_k = top_k*larger
        # print('top_k ',top_k.shape)
        # print('top_k ',top_k)
        att_1 = att_1*top_k
        #print('att_1 ',att_1.shape)
        # assert 1==2
        att_2 = att_1.unsqueeze(2).repeat(1,1,128)
        Es = selected_embeddings*att_2
        return Es
    
    def orcal_EE(self, x, embedding, label):
        batch, time, dim = x.shape

        mixture_embedding = self.encoder(x) # 8, 125, 128
        mixture_embedding = mixture_embedding.transpose(1,2)
        mixture_embedding = self.bn(mixture_embedding)
        mixture_embedding = mixture_embedding.transpose(1,2)

        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding_pre = embedding.unsqueeze(1)
        embedding_pre = embedding_pre.repeat(1, x.shape[1], 1)
        f = self.detection.fusion(embedding_pre, x) # the first stage results
        #f = torch.cat((x, embedding_pre), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        f, _ = self.detection.gru(f) #  x  torch.Size([16, 125, 256])
        f = self.detection.fc(f)
        decision_time = torch.softmax(self.detection.outputlayer(f),dim=2) # x  torch.Size([16, 125, 2])
        
        selected_embeddings, top_k = self.select_topk_embeddings(decision_time[:,:,0], mixture_embedding, self.top)
        
        selected_embeddings = self.sum_with_attention(embedding, top_k, selected_embeddings) # add the weight

        mix_embedding = selected_embeddings.mean(1).unsqueeze(1) # 
        mix_embedding = mix_embedding.repeat(1, x.shape[1], 1)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        mix_embedding = self.EE_fusion(mix_embedding, embedding) # 使用神经网络进行融合
        # mix_embedding2 = selected_embeddings2.mean(1)
        #mix_embedding =  embedding + mix_embedding # 直接相加
        # new detection results
        # embedding_now = mix_embedding.unsqueeze(1)
        # embedding_now = embedding_now.repeat(1, x.shape[1], 1)
        f_now = self.detection.fusion(mix_embedding, x) 
        #f_now = torch.cat((x, embedding_now), dim=2) # 
        f_now, _ = self.detection.gru(f_now) #  x  torch.Size([16, 125, 256])
        f_now = self.detection.fc(f_now)
        decision_time_now = torch.softmax(self.detection.outputlayer(f_now), dim=2) # x  torch.Size([16, 125, 2])
        
        top_k = top_k.mean(1)  # get avg score,higher score will have more weight
        larger = top_k > self.tao
        top_k = top_k * larger
        top_k = top_k/2.0
        # print('top_k ',top_k)
        # assert 1==2
        # print('tok_k[ ',top_k.shape)
        # print('decision_time ',decision_time.shape)
        # print('decision_time_now ',decision_time_now.shape)
        neg_w = top_k.unsqueeze(1).unsqueeze(2)
        neg_w = neg_w.repeat(1, decision_time_now.shape[1], decision_time_now.shape[2])
        # print('neg_w ',neg_w.shape)
        #print('neg_w ',neg_w[:,0:10,0])
        pos_w = 1-neg_w
        #print('pos_w ',pos_w[:,0:10,0])
        decision_time_final = decision_time*pos_w + neg_w*decision_time_now
        #print('decision_time_final ',decision_time_final[0,0:10,0])
        # print(decision_time_final[0,:,:])
        #assert 1==2
        return decision_time_final
    
    def forward(self, x, ref, label=None):
        batch, time, dim = x.shape
        logit = torch.zeros(1).cuda()
        embeddings  = self.encoder(ref)
        mean_embedding = embeddings.mean(1)
        if self.att_pool == True:
            mean_embedding = self.bn(mean_embedding)
            embeddings = embeddings.transpose(1,2)
            embeddings = self.bn(embeddings)
            embeddings = embeddings.transpose(1,2)
            embedding = self.attention_pooling(embeddings, mean_embedding)
        else:
            embedding = mean_embedding
        if self.enhancement == True:
            decision_time = self.orcal_EE(x, embedding, label)
            decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
            return decision_time[:,:,0], decision_up, logit

        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        # x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        x = self.detection.fusion(embedding, x) 
        # embedding = embedding.unsqueeze(1)
        # embedding = embedding.repeat(1, x.shape[1], 1)
        # x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        x, _ = self.detection.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.detection.fc(x)
        decision_time = torch.softmax(self.detection.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2),
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0], decision_up, logit

class RaDur_fusion_sub(nn.Module):
    def __init__(self, model_config, inputdim, outputdim, time_resolution, **kwargs):
        super().__init__()
        self.encoder = Cnn14()
        self.detection = CDur_CNN_mul_scale_fusion(inputdim, outputdim, time_resolution)
        self.softmax = nn.Softmax(dim=2)
        #self.temperature = 5
        if model_config['pre_train']:
            self.encoder.load_state_dict(torch.load(model_config['encoder_path'])['model'])
            self.detection.load_state_dict(torch.load(model_config['CDur_path']))
        self.q = nn.Linear(128,128)
        self.k = nn.Linear(128,128)
        self.q_ee = nn.Linear(128, 128)
        self.k_ee = nn.Linear(128, 128)
        self.temperature = 11.3 # sqrt(128)
        self.att_pool = model_config['att_pool']
        self.enhancement = model_config['enhancement'] 
        self.tao = model_config['tao']
        self.top = model_config['top']
        self.s_tao = model_config['s_tao']
        self.bn = nn.BatchNorm1d(128)
        self.EE_fusion = Fusion(128,4)

    def get_w(self,q,k):
        q = self.q(q)
        k = self.k(k)
        q = q.unsqueeze(1)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn/self.temperature
        attn = self.softmax(attn)
        return attn
    
    def get_w_ee(self,q,k):
        q = self.q_ee(q)
        k = self.k_ee(k)
        q = q.unsqueeze(1)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn/self.temperature
        attn = self.softmax(attn)
        return attn
    
    def attention_pooling(self, embeddings, mean_embedding):
        att_pool_w = self.get_w(mean_embedding,embeddings)
        embedding = torch.bmm(att_pool_w, embeddings).squeeze(1)
        return embedding
    
    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1) # 根据分数进行排序
        top_k = _[:,:k]
        top_k = top_k.mean(1)
        idx_topk = idx_DESC[:, :k] # 取top_k个
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings,top_k
    
    def sum_with_attention(self, embedding, top_k, selected_embeddings):
        # print('embedding ',embedding)
        # print('selected_embeddings ',selected_embeddings.shape)
        att_1 = self.get_w_ee(embedding, selected_embeddings)
        att_1 = att_1.squeeze(1)
        #print('att_1 ',att_1.shape)
        larger = top_k > self.tao
        # print('larger ',larger)
        top_k = top_k*larger
        # print('top_k ',top_k.shape)
        # print('top_k ',top_k)
        att_1 = att_1*top_k
        #print('att_1 ',att_1.shape)
        # assert 1==2
        att_2 = att_1.unsqueeze(2).repeat(1,1,128)
        Es = selected_embeddings*att_2
        return Es
    
    def orcal_EE(self, x, embedding, label):
        batch, time, dim = x.shape

        mixture_embedding = self.encoder(x) # 8, 125, 128
        mixture_embedding = mixture_embedding.transpose(1,2)
        mixture_embedding = self.bn(mixture_embedding)
        mixture_embedding = mixture_embedding.transpose(1,2)

        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding_pre = embedding.unsqueeze(1)
        embedding_pre = embedding_pre.repeat(1, x.shape[1], 1)
        f = self.detection.fusion(embedding_pre, x) # the first stage results
        #f = torch.cat((x, embedding_pre), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        f, _ = self.detection.gru(f) #  x  torch.Size([16, 125, 256])
        f = self.detection.fc(f)
        decision_time = torch.softmax(self.detection.outputlayer(f),dim=2) # x  torch.Size([16, 125, 2])
        
        selected_embeddings, top_k = self.select_topk_embeddings(decision_time[:,:,0], mixture_embedding, self.top)
        #selected_embeddings, top_k  = self.select_topk_embeddings(label, mixture_embedding, self.top)
        
        selected_embeddings = self.sum_with_attention(embedding, top_k, selected_embeddings) # add the weight

        mix_embedding = selected_embeddings.mean(1).unsqueeze(1) # 
        mix_embedding = mix_embedding.repeat(1, x.shape[1], 1)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        mix_embedding = self.EE_fusion(mix_embedding, embedding) # 使用神经网络进行融合
        f_now = self.detection.fusion(mix_embedding, x) 
        #f_now = torch.cat((x, embedding_now), dim=2) # 
        f_now, _ = self.detection.gru(f_now) #  x  torch.Size([16, 125, 256])
        f_now = self.detection.fc(f_now)
        decision_time_now = torch.softmax(self.detection.outputlayer(f_now), dim=2) # x  torch.Size([16, 125, 2])
        
        top_k = top_k.mean(1)  # get avg score,higher score will have more weight
        larger = top_k > self.tao # 大于thres 相加，
        smaller = top_k < self.s_tao # 小于直接减？
        top_k_add = top_k * larger
        top_k_sub = -1.0*top_k*smaller
        tok_k = top_k_add + top_k_sub
        top_k = top_k/2.0
        # print('top_k ',top_k)
        neg_w = top_k.unsqueeze(1).unsqueeze(2)
        neg_w = neg_w.repeat(1, decision_time_now.shape[1], decision_time_now.shape[2])
        
        pos_w = 1-neg_w
        pos_w = pos_w.clamp(0,1)
        #print('pos_w ',pos_w[:,0:10,0])
        decision_time_final = decision_time*pos_w + neg_w*decision_time_now
        return decision_time_final
    
    def forward(self, x, ref, label=None):
        batch, time, dim = x.shape
        logit = torch.zeros(1).cuda()
        embeddings  = self.encoder(ref)
        mean_embedding = embeddings.mean(1)
        if self.att_pool == True:
            mean_embedding = self.bn(mean_embedding)
            embeddings = embeddings.transpose(1,2)
            embeddings = self.bn(embeddings)
            embeddings = embeddings.transpose(1,2)
            embedding = self.attention_pooling(embeddings, mean_embedding)
        else:
            embedding = mean_embedding
        if self.enhancement == True:
            decision_time = self.orcal_EE(x, embedding, label)
            decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
            return decision_time[:,:,0], decision_up, logit

        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        # x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        x = self.detection.fusion(embedding, x) 
        # embedding = embedding.unsqueeze(1)
        # embedding = embedding.repeat(1, x.shape[1], 1)
        # x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        x, _ = self.detection.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.detection.fc(x)
        decision_time = torch.softmax(self.detection.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2),
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0], decision_up, logit


class S_student(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = Cnn10()
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(640, 512, bidirectional=True, batch_first=True)
        # self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, outputdim)
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
    def forward(self, x, embedding,one_hot=None): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class CDur_CNN14_2GRU(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = Cnn10()
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(640, 512,2, bidirectional=True, batch_first=True) # previous is 640
        # self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, outputdim)
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
    def forward(self, x, embedding,one_hot=None): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)

        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class CDur_CNN14_and_one(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = Cnn10()
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(896, 512, bidirectional=True, batch_first=True) # previous is 640
        # self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, outputdim)
        self.embed_layer = nn.Embedding(192,256) # encode 
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
    def forward(self, x, embedding,one_hot=None): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)

        one_hot_embed = self.embed_layer(one_hot)
        #print('one_hot_embed',one_hot_embed.shape)
        one_hot_embed = one_hot_embed.unsqueeze(1)
        one_hot_embed = one_hot_embed.repeat(1, x.shape[1], 1)

        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        x = torch.cat((x,one_hot_embed),dim=2) # 
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class CDur_CNN14_one_hot(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = Cnn10()
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(640+128, 512, bidirectional=True, batch_first=True)
        # self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, outputdim)
        self.embed_layer = nn.Embedding(192,256) # encode 
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
    def forward(self, x, embedding,one_hot=None): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        one_hot_embed = self.embed_layer(one_hot)
        #print('one_hot_embed',one_hot_embed.shape)
        one_hot_embed = one_hot_embed.unsqueeze(1)
        one_hot_embed = one_hot_embed.repeat(1, x.shape[1], 1)
        # embedding = embedding.unsqueeze(1)
        # embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, one_hot_embed), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class CDur_CNN14_fusion(nn.Module):
    def __init__(self, inputdim, outputdim,time_resolution, **kwargs):
        super().__init__()
        self.features = Cnn10()
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(512, 512, bidirectional=True, batch_first=True)
        # self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.fusion = Fusion(128,512,2)
        self.outputlayer = nn.Linear(256, outputdim)
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
    def forward(self, x, embedding, one_hot=None): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = self.fusion(embedding, x)
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class CDur_CNN14_my(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = Cnn10()
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        # self.gru = nn.GRU(640, 512, bidirectional=True, batch_first=True)
        self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, outputdim)
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, embedding,one_hot=None): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        # if not hasattr(self, '_flattened'):
        #     self.gru.flatten_parameters()
        # x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class Join_fake(nn.Module):
    def __init__(self,model_config,inputdim,outputdim,time_resolution,**kwargs):
        super().__init__()
        self.encoder = Cnn14()
        self.detection = CDur_CNN14(inputdim,outputdim,time_resolution,**kwargs)
        self.softmax = nn.Softmax(dim=2)
        self.temperature = 5
        if model_config['pre_train']:
            self.encoder.load_state_dict(torch.load(model_config['encoder_path'])['model'])
        if model_config['CDur_pretrain']:
            self.detection.load_state_dict(torch.load(model_config['CDur_path']))
        for p in self.encoder.parameters(): # fix the parameter of w_student
            p.requires_grad = False
        self.enhancement = model_config['enhancement']
        self.att_pool = model_config['attention_pooling']
    def get_w(self,q,k):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn/self.temperature
        attn = self.softmax(attn)
        return attn
    
    def attention_pooling(self,embeddings,mean_embedding):
        att_pool_w = self.get_w(mean_embedding,embeddings)
        embedding = torch.bmm(att_pool_w, embeddings).squeeze(1)
        return embedding
    
    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1) # 根据分数进行排序
        top_k = _[:,:k]
        idx_topk = idx_DESC[:, :k] # 取top_k个
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings,top_k
    
    def orcal_EE2(self,x,embedding,label):
        batch, time, dim = x.shape
        mixture_embedding = self.encoder(x) # 8, 125, 128
        for i in range(batch):
            index_ = label[i,:] > 0.5
            tmp_mix_embed = mixture_embedding[i,index_,:]
            if tmp_mix_embed.shape[0]>0:
                embedding[i,:] = tmp_mix_embed.mean(0)
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding_pre = embedding.unsqueeze(1)
        embedding_pre = embedding_pre.repeat(1, x.shape[1], 1)
        f = torch.cat((x, embedding_pre), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        f, _ = self.detection.gru(f) #  x  torch.Size([16, 125, 256])
        f = self.detection.fc(f)
        decision_time = torch.softmax(self.detection.outputlayer(f),dim=2) # x  torch.Size([16, 125, 2])
        return decision_time

    def orcal_EE(self,x,embedding,label):
        batch, time, dim = x.shape
        mixture_embedding = self.encoder(x) # 8, 125, 128
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding_pre = embedding.unsqueeze(1)
        embedding_pre = embedding_pre.repeat(1, x.shape[1], 1)
        f = torch.cat((x, embedding_pre), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        f, _ = self.detection.gru(f) #  x  torch.Size([16, 125, 256])
        f = self.detection.fc(f)
        decision_time = torch.softmax(self.detection.outputlayer(f),dim=2) # x  torch.Size([16, 125, 2])
        psedu_label = decision_time[:,:,0]*label
        # print('decision_time ',decision_time)
        # assert 1==2
        selected_embeddings, top_k  = self.select_topk_embeddings(psedu_label, mixture_embedding, 10)
        mix_embedding = selected_embeddings.mean(1) # 
        # mix_embedding2 = selected_embeddings2.mean(1)
        mix_embedding =  embedding + mix_embedding
        # new detection results
        embedding_now = mix_embedding.unsqueeze(1)
        embedding_now = embedding_now.repeat(1, x.shape[1], 1)
        f_now = torch.cat((x, embedding_now), dim=2) # 
        f_now, _ = self.detection.gru(f_now) #  x  torch.Size([16, 125, 256])
        f_now = self.detection.fc(f_now)
        decision_time_now = torch.softmax(self.detection.outputlayer(f_now),dim=2) # x  torch.Size([16, 125, 2])
        # print('decision_time ',decision_time_now[:,:,0])
        # assert 1==2
        # print('decision_time_now ',decision_time_now[0,0:10,0])
        top_k = top_k.mean(1)  # get avg score,higher score will have more weight
        larger = top_k > 0.1
        top_k = top_k * larger
        top_k = top_k/2.0
        neg_w = top_k.unsqueeze(1).unsqueeze(2)
        neg_w = neg_w.repeat(1,decision_time_now.shape[1],decision_time_now.shape[2])
        pos_w = 1-neg_w
        decision_time_final = decision_time*pos_w + neg_w*decision_time_now
        return decision_time_final


    def embedding_enhancement_2(self,x,embedding):
        # In this part, we donot add select embedding with real embedding, rather then we use vote methods
        batch, time, dim = x.shape
        mixture_embedding = self.encoder(x) # 8, 125, 128
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding_pre = embedding.unsqueeze(1)
        embedding_pre = embedding_pre.repeat(1, x.shape[1], 1)
        f = torch.cat((x, embedding_pre), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        f, _ = self.detection.gru(f) #  x  torch.Size([16, 125, 256])
        f = self.detection.fc(f)
        decision_time = torch.softmax(self.detection.outputlayer(f),dim=2) # x  torch.Size([16, 125, 2])
        selected_embeddings,top_k = self.select_topk_embeddings(decision_time[:,:,0],mixture_embedding,2)
        # embedding_tmp = embedding.unsqueeze(1)
        # att_w = self.get_w(embedding_tmp,selected_embeddings)
        #mix_embedding = torch.bmm(att_w, selected_embeddings).squeeze(1)
        mix_embedding = selected_embeddings.mean(1) # 
        mix_embedding = mix_embedding + embedding
        # new detection results
        embedding_now = mix_embedding.unsqueeze(1)
        embedding_now = embedding_now.repeat(1,x.shape[1],1)
        f_now = torch.cat((x,embedding_now),dim=2) # 
        f_now, _ = self.detection.gru(f_now) #  x  torch.Size([16, 125, 256])
        f_now = self.detection.fc(f_now)
        decision_time_now = torch.softmax(self.detection.outputlayer(f_now),dim=2) # x  torch.Size([16, 125, 2])
        # print('decision_time ',decision_time[0,0:10,0])
        # print('decision_time_now ',decision_time_now[0,0:10,0])
        # assert 1==2
        # merge the predict results
        #print('top_k ', top_k.mean(1))
        top_k = top_k.mean(1)  # get avg score,higher score will have more weight
        larger = top_k > 0.8
        top_k = top_k* larger
        top_k = top_k/2.0
        # print('top_k ',top_k)
        # assert 1==2
        # print('tok_k[ ',top_k.shape)
        # print('decision_time ',decision_time.shape)
        # print('decision_time_now ',decision_time_now.shape)
        neg_w = top_k.unsqueeze(1).unsqueeze(2)
        neg_w = neg_w.repeat(1,decision_time_now.shape[1],decision_time_now.shape[2])
        # print('neg_w ',neg_w.shape)
        #print('neg_w ',neg_w[:,0:10,0])
        pos_w = 1-neg_w
        #print('pos_w ',pos_w[:,0:10,0])
        decision_time_final = decision_time*pos_w + neg_w*decision_time_now
        #print('decision_time_final ',decision_time_final[0,0:10,0])
        # print(decision_time_final[0,:,:])
        #assert 1==2
        return decision_time_final


    def embedding_enhancement(self,x,embedding):
        batch, time_, dim = x.shape
        with torch.no_grad():
            mixture_embedding = self.encoder(x) # 8, 125, 128
            x = x.unsqueeze(1) # (b,1,t,d) 
            x = self.detection.features(x) # 
            x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
            embedding_pre = embedding.unsqueeze(1)
            embedding_pre = embedding_pre.repeat(1, x.shape[1], 1)
            x = torch.cat((x, embedding_pre), dim=2) # [B, T, 128 + emb_dim]
            if not hasattr(self, '_flattened'):
                self.detection.gru.flatten_parameters()
            x, _ = self.detection.gru(x) #  x  torch.Size([16, 125, 256])
            x = self.detection.fc(x)
            decision_time = torch.softmax(self.detection.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        selected_embeddings,top_k = self.select_topk_embeddings(decision_time[:,:,0], mixture_embedding, 2)
        # print('selected_embeddings ',selected_embeddings.shape)
        # print('top_k ',top_k.shape)
        # print('top_k ',top_k)
        # print('index_top ',index_top)
        for i in range(batch):
            index_top = top_k[i,:] > 0.75
            #print('index_top ',index_top.shape)
            selected_embeddings_tmp = selected_embeddings[i,index_top,:]
            #print('selected_embeddings ',selected_embeddings_tmp.shape)
            if selected_embeddings_tmp.shape[0]==0:
                continue
            selected_embeddings_tmp = selected_embeddings_tmp.unsqueeze(0)
            embedding_tmp = embedding[i,:].unsqueeze(0).unsqueeze(1)
            att_w = self.get_w(embedding_tmp,selected_embeddings_tmp)
            # print('att_w ',att_w.shape)
            # print(att_w)
            mix_embedding = torch.bmm(att_w, selected_embeddings_tmp).squeeze(1)
            # print('mix_embedding ',mix_embedding.shape)
            # assert 1==2
            embedding[i,:] = embedding[i,:] + 0.3*mix_embedding
        return embedding

        
        # embedding_tmp = embedding.unsqueeze(1)
        # print('embedding_tmp ',embedding_tmp.shape)
        # print('selected_embeddings ',selected_embeddings.shape)
        # att_w = self.get_w(embedding_tmp,selected_embeddings)
        # mix_embedding = torch.bmm(att_w, selected_embeddings).squeeze(1)
        # assert 1==2
        # print('mix_embedding ',mix_embedding.shape)
        # assert 1==2
        # att = top_k-0.55
        # # print('att ',att)
        # att_0 = top_k > 0.55
        # # print('att_0 ',att_0)
        # att = att_0 * att
        # # print('att ',att.shape)
        # att = att.unsqueeze(1)
        # att = att.repeat(1,mix_embedding.shape[1])
        # print(att.shape)
        # print(att[0])
        # assert 1 == 2
        #final_embedding = embedding + 0.1*mix_embedding
        # if top_k.all() < 0.55:
        #     final_embedding = embedding
        # else:
        #     final_embedding = embedding + 0.1*mix_embedding
        #return final_embedding

    def forward(self,x,ref,label=None):
        batch, time_, dim = x.shape
        logit = torch.zeros(1).cuda()
        embeddings  = self.encoder(ref)
        mean_embedding = embeddings.mean(1).unsqueeze(1)
        if self.att_pool == True:
            embedding = self.attention_pooling(embeddings,mean_embedding)
        else:
            embedding = mean_embedding.squeeze(1)
        if self.enhancement == True:
            # decision_time = self.embedding_enhancement_2(x,embedding)
            decision_time = self.orcal_EE(x, embedding, label)
            decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2),
                time_, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
            return decision_time[:,:,0],decision_up,logit
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        x, _ = self.detection.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.detection.fc(x)
        decision_time = torch.softmax(self.detection.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        # if enhancement:
        #     self.memory_enhancement(x_enhance,decision_time)
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2),
                time_, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up,logit



class Join(nn.Module):
    def __init__(self,model_config,inputdim,outputdim,time_resolution,**kwargs):
        super().__init__()
        self.encoder = Cnn14()
        self.detection = CDur_CNN14(inputdim,outputdim,time_resolution,**kwargs)
        self.softmax = nn.Softmax(dim=2)
        self.temperature = 5
        #self.memory_bank = np.zeros((192,128))
        #self.spk_emb_file_path = model_config['spk_emb_file_path']
        #self.init_memory()
        if model_config['pre_train']:
            self.encoder.load_state_dict(torch.load(model_config['encoder_path'])['model'])
        if model_config['CDur_pretrain']:
            self.detection.load_state_dict(torch.load(model_config['CDur_path']))
        self.enhancement = model_config['enhancement']
        self.att_pool = model_config['attention_pooling']
        self.top = 3
        self.lamda = 0.9
    # def init_memory(self):
    #     X = []
    #     with open(self.spk_emb_file_path, 'r') as file:
    #         for line in file:
    #             temp_line = line.strip().split('\t')
    #             file_id = os.path.basename(temp_line[0]) # get filename
    #             emb = np.array(temp_line[1].split(' ')).astype(np.float) # embedding
    #             X.append(emb)
    #     X = np.array(X)
    #     kmeans = KMeans(n_clusters=192, random_state=0).fit(X)
    #     label_ = kmeans.labels_
    #     for i in range(192): # init self.memory_bank
    #         index_ = (label_ == i)
    #         self.memory_bank[i,:] = X[index_,:].mean(0)
    # def get_dict(self,label_):
    #     query_by_label = {}
    #     for i,lb in enumerate(label_):
    #         query_by_label[lb] = i
    #     return query_by_label
    def get_w(self,q,k):
        q = q.unsqueeze(1)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn/self.temperature
        attn = self.softmax(attn)
        return attn
    
    def attention_pooling(self,embeddings,mean_embedding):
        att_pool_w = self.get_w(mean_embedding,embeddings)
        embedding = torch.bmm(att_pool_w, embeddings).squeeze(1)
        # print(embedding.shape)
        # print(att_pool_w.shape)
        # print(att_pool_w[0])
        # assert 1==2
        return embedding
    
    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1) # 根据分数进行排序
        top_k = _[:,:k]
        # print('top_k ', top_k)
        # top_k = top_k.mean(1)
        idx_topk = idx_DESC[:, :k] # 取top_k个
        # print('index ', idx_topk)
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings,top_k
    
    # def memory_enhancement(self,x,decision_time):
    #     batch, time_, dim = x.shape
    #     mixture_embedding = self.encoder(x) # 8, 125, 128
    #     selected_embeddings,top_k = self.select_topk_embeddings(decision_time[:,:,0],mixture_embedding,1)
    #     # print('top_k ',top_k.shape)
    #     # assert 1==2
    #     # if top_k.min() < 0.8:
    #     #     return
    #     index_top = top_k > 0.8
    #     # print('top_k ',top_k)
    #     # print('index_top ',index_top)
    #     selected_embeddings = selected_embeddings[index_top,:]
    #     # print('selected_embeddings ',selected_embeddings.shape)
    #     # assert 1==2
    #     if selected_embeddings.shape[0]==0:
    #         return
    #     selected_embeddings_np = selected_embeddings.detach().cpu().numpy()
    #     x_tmp = np.concatenate((self.memory_bank,selected_embeddings_np[:,0,:]),axis=0)
    #     #np_embedding = embedding.detach().cpu().numpy()
    #     # st_time = time.time()
    #     kmeans = KMeans(n_clusters=192, random_state=0, n_init=5).fit(x_tmp)
    #     # ed_time = time.time()
    #     label_ = kmeans.labels_
    #     for i in range(192): # init self.memory_bank
    #         index_ = (label_ == i)
    #         if np.all(index_==False):
    #             self.memory_bank[i,:] = x_tmp[i,:]
    #         else:
    #             self.memory_bank[i,:] = x_tmp[index_,:].mean(0)

    def orcal_EE2(self,x,embedding,label):
        batch, time, dim = x.shape
        mixture_embedding = self.encoder(x) # 8, 125, 128
        for i in range(batch):
            index_ = label[i,:] > 0.5
            # print(' label[i,:] ', label[i,:])
            # print('index_ ',index_)
            tmp_mix_embed = mixture_embedding[i,index_,:]
            # print('tmp_mix_embed ',tmp_mix_embed.shape)
            # assert 1==2
            if tmp_mix_embed.shape[0]>0:
                embedding[i,:] = tmp_mix_embed.mean(0)
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding_pre = embedding.unsqueeze(1)
        embedding_pre = embedding_pre.repeat(1, x.shape[1], 1)
        f = torch.cat((x, embedding_pre), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        f, _ = self.detection.gru(f) #  x  torch.Size([16, 125, 256])
        f = self.detection.fc(f)
        decision_time = torch.softmax(self.detection.outputlayer(f),dim=2) # x  torch.Size([16, 125, 2])
        return decision_time

    def orcal_EE(self,x,embedding,label):
        batch, time, dim = x.shape
        mixture_embedding = self.encoder(x) # 8, 125, 128
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding_pre = embedding.unsqueeze(1)
        embedding_pre = embedding_pre.repeat(1, x.shape[1], 1)
        f = torch.cat((x, embedding_pre), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        f, _ = self.detection.gru(f) #  x  torch.Size([16, 125, 256])
        f = self.detection.fc(f)
        decision_time = torch.softmax(self.detection.outputlayer(f),dim=2) # x  torch.Size([16, 125, 2])
        # print('decision ........')
        # print(decision_time[:,:,0])
        # print('_ ',_[0])
        # print('idx_DESC_w ', idx_DESC[0])
        # print('att_weight ', att_weight[0])
        # print('label ', label)
        # _, index_desc = label.sort(descending=True,dim=1)
        # # print('label index_desc ', index_desc)
        # # print('decision_time ', decision_time[:,:,0])
        # _, index_desc = decision_time[:,:,0].sort(descending=True,dim=1)
        # print('index_desc_decision ',index_desc)
        # selected_embeddings, top_k = self.select_topk_embeddings(decision_time[:,:,0], mixture_embedding, 30)
        # print('decision_time ',decision_time.shape)
        # print('label ',label.shape)
        # print(label)
        psedu_label = decision_time[:,:,0]*label
        # print('decision_time ',decision_time)
        # assert 1==2
        selected_embeddings, top_k  = self.select_topk_embeddings(psedu_label, mixture_embedding, 10)
        # print('label ........')
        # new_decision_score, _ = decision_time[:,:,0].sort(descending=True,dim=1)
        # new_decision_score = new_decision_score[:,:30]
        # embedding_tmp = embedding.unsqueeze(1)
        # att_weight = self.get_w(embedding_tmp,selected_embeddings)
        # att_weight = att_weight.squeeze(1)
        # final_select_emb, _ = self.select_topk_embeddings(att_weight,selected_embeddings,2)
        # print('att_weight.shape ', att_weight.shape)
        # _, idx_DESC = att_weight.sort(descending=True, dim=1)
        # idx_DESC = idx_DESC[:,:2]
        # top_k = new_decision_score[:,idx_DESC.squeeze()]
        # print(label)
        # selected_embeddings,top_k = self.select_topk_embeddings(label, mixture_embedding, 2)
        # assert 1==2
        mix_embedding = selected_embeddings.mean(1) # 
        # mix_embedding2 = selected_embeddings2.mean(1)
        mix_embedding =  embedding + mix_embedding
        # new detection results
        embedding_now = mix_embedding.unsqueeze(1)
        embedding_now = embedding_now.repeat(1, x.shape[1], 1)
        f_now = torch.cat((x, embedding_now), dim=2) # 
        f_now, _ = self.detection.gru(f_now) #  x  torch.Size([16, 125, 256])
        f_now = self.detection.fc(f_now)
        decision_time_now = torch.softmax(self.detection.outputlayer(f_now),dim=2) # x  torch.Size([16, 125, 2])
        # print('decision_time ',decision_time_now[:,:,0])
        # assert 1==2
        # print('decision_time_now ',decision_time_now[0,0:10,0])
        top_k = top_k.mean(1)  # get avg score,higher score will have more weight
        larger = top_k > 0.1
        top_k = top_k * larger
        top_k = top_k/2.0
        # print('top_k ',top_k)
        # assert 1==2
        # print('tok_k[ ',top_k.shape)
        # print('decision_time ',decision_time.shape)
        # print('decision_time_now ',decision_time_now.shape)
        neg_w = top_k.unsqueeze(1).unsqueeze(2)
        neg_w = neg_w.repeat(1,decision_time_now.shape[1],decision_time_now.shape[2])
        # print('neg_w ',neg_w.shape)
        #print('neg_w ',neg_w[:,0:10,0])
        pos_w = 1-neg_w
        #print('pos_w ',pos_w[:,0:10,0])
        decision_time_final = decision_time*pos_w + neg_w*decision_time_now
        #print('decision_time_final ',decision_time_final[0,0:10,0])
        # print(decision_time_final[0,:,:])
        #assert 1==2
        return decision_time_final


    def embedding_enhancement_2(self,x,embedding):
        # In this part, we donot add select embedding with real embedding, rather then we use vote methods
        batch, time, dim = x.shape
        mixture_embedding = self.encoder(x) # 8, 125, 128
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding_pre = embedding.unsqueeze(1)
        embedding_pre = embedding_pre.repeat(1, x.shape[1], 1)
        f = torch.cat((x, embedding_pre), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        f, _ = self.detection.gru(f) #  x  torch.Size([16, 125, 256])
        f = self.detection.fc(f)
        decision_time = torch.softmax(self.detection.outputlayer(f),dim=2) # x  torch.Size([16, 125, 2])
        selected_embeddings,top_k = self.select_topk_embeddings(decision_time[:,:,0],mixture_embedding,2)
        # embedding_tmp = embedding.unsqueeze(1)
        # att_w = self.get_w(embedding_tmp,selected_embeddings)
        #mix_embedding = torch.bmm(att_w, selected_embeddings).squeeze(1)
        mix_embedding = selected_embeddings.mean(1) # 
        mix_embedding = mix_embedding + embedding
        # new detection results
        embedding_now = mix_embedding.unsqueeze(1)
        embedding_now = embedding_now.repeat(1,x.shape[1],1)
        f_now = torch.cat((x,embedding_now),dim=2) # 
        f_now, _ = self.detection.gru(f_now) #  x  torch.Size([16, 125, 256])
        f_now = self.detection.fc(f_now)
        decision_time_now = torch.softmax(self.detection.outputlayer(f_now),dim=2) # x  torch.Size([16, 125, 2])
        # print('decision_time ',decision_time[0,0:10,0])
        # print('decision_time_now ',decision_time_now[0,0:10,0])
        # assert 1==2
        # merge the predict results
        #print('top_k ', top_k.mean(1))
        top_k = top_k.mean(1)  # get avg score,higher score will have more weight
        larger = top_k > 0.8
        top_k = top_k* larger
        top_k = top_k/2.0
        # print('top_k ',top_k)
        # assert 1==2
        # print('tok_k[ ',top_k.shape)
        # print('decision_time ',decision_time.shape)
        # print('decision_time_now ',decision_time_now.shape)
        neg_w = top_k.unsqueeze(1).unsqueeze(2)
        neg_w = neg_w.repeat(1,decision_time_now.shape[1],decision_time_now.shape[2])
        # print('neg_w ',neg_w.shape)
        #print('neg_w ',neg_w[:,0:10,0])
        pos_w = 1-neg_w
        #print('pos_w ',pos_w[:,0:10,0])
        decision_time_final = decision_time*pos_w + neg_w*decision_time_now
        #print('decision_time_final ',decision_time_final[0,0:10,0])
        # print(decision_time_final[0,:,:])
        #assert 1==2
        return decision_time_final


    def embedding_enhancement(self,x,embedding):
        batch, time_, dim = x.shape
        with torch.no_grad():
            mixture_embedding = self.encoder(x) # 8, 125, 128
            x = x.unsqueeze(1) # (b,1,t,d) 
            x = self.detection.features(x) # 
            x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
            embedding_pre = embedding.unsqueeze(1)
            embedding_pre = embedding_pre.repeat(1, x.shape[1], 1)
            x = torch.cat((x, embedding_pre), dim=2) # [B, T, 128 + emb_dim]
            if not hasattr(self, '_flattened'):
                self.detection.gru.flatten_parameters()
            x, _ = self.detection.gru(x) #  x  torch.Size([16, 125, 256])
            x = self.detection.fc(x)
            decision_time = torch.softmax(self.detection.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        selected_embeddings,top_k = self.select_topk_embeddings(decision_time[:,:,0], mixture_embedding, 2)
        # print('selected_embeddings ',selected_embeddings.shape)
        # print('top_k ',top_k.shape)
        # print('top_k ',top_k)
        # print('index_top ',index_top)
        for i in range(batch):
            index_top = top_k[i,:] > 0.75
            #print('index_top ',index_top.shape)
            selected_embeddings_tmp = selected_embeddings[i,index_top,:]
            #print('selected_embeddings ',selected_embeddings_tmp.shape)
            if selected_embeddings_tmp.shape[0]==0:
                continue
            selected_embeddings_tmp = selected_embeddings_tmp.unsqueeze(0)
            embedding_tmp = embedding[i,:].unsqueeze(0).unsqueeze(1)
            att_w = self.get_w(embedding_tmp,selected_embeddings_tmp)
            # print('att_w ',att_w.shape)
            # print(att_w)
            mix_embedding = torch.bmm(att_w, selected_embeddings_tmp).squeeze(1)
            # print('mix_embedding ',mix_embedding.shape)
            # assert 1==2
            embedding[i,:] = embedding[i,:] + 0.3*mix_embedding
        return embedding

        
        # embedding_tmp = embedding.unsqueeze(1)
        # print('embedding_tmp ',embedding_tmp.shape)
        # print('selected_embeddings ',selected_embeddings.shape)
        # att_w = self.get_w(embedding_tmp,selected_embeddings)
        # mix_embedding = torch.bmm(att_w, selected_embeddings).squeeze(1)
        # assert 1==2
        # print('mix_embedding ',mix_embedding.shape)
        # assert 1==2
        # att = top_k-0.55
        # # print('att ',att)
        # att_0 = top_k > 0.55
        # # print('att_0 ',att_0)
        # att = att_0 * att
        # # print('att ',att.shape)
        # att = att.unsqueeze(1)
        # att = att.repeat(1,mix_embedding.shape[1])
        # print(att.shape)
        # print(att[0])
        # assert 1 == 2
        #final_embedding = embedding + 0.1*mix_embedding
        # if top_k.all() < 0.55:
        #     final_embedding = embedding
        # else:
        #     final_embedding = embedding + 0.1*mix_embedding
        #return final_embedding

    def EE(self, x, embedding, label=None):
        batch, time, dim = x.shape
        mixture_embedding = self.encoder(x) # 8, 125, 128
        # print('label ',label.shape)
        # assert 1==2
        selected_embeddings, top_k  = self.select_topk_embeddings(label, mixture_embedding, self.top)
        # print('embedding ',embedding.shape)
        # print('selected_embeddings ',selected_embeddings.shape)
        # print('top_k ', top_k)
        top_k_pre = top_k
        top_k_label = torch.ones_like(top_k).cuda()
        # bceLoss = BCELoss()
        # loss_top = bceLoss(top_k_pre,top_k_label)
        att_1 = self.get_w(embedding, selected_embeddings)
        att_1 = att_1.squeeze()
        # print('att_1 ',att_1)
        larger = top_k > 0.5
        # print('larger ',larger)
        top_k = top_k*larger
        # print('top_k ',top_k)
        att_1 = att_1*top_k
        # print('att_1 ',att_1)
        att_2 = att_1.unsqueeze(2).repeat(1,1,128)
        # print('att_2 ',att_2.shape)
        # print('att_2 ',att_2[:,:,:10])
        Es = selected_embeddings*att_2
        E = Es.mean(1)
        # print('E ',E.shape)
        # assert 1==2
        final_embedding = self.lamda*embedding + (1-self.lamda)*E
        return final_embedding

    def forward(self,x,ref,label=None):
        batch, time_, dim = x.shape
        logit = torch.zeros(1).cuda()
        embeddings = self.encoder(ref)
        mean_embedding = embeddings.mean(1)
        if self.att_pool == True:
            embedding = self.attention_pooling(embeddings,mean_embedding)
        else:
            embedding = mean_embedding
        if self.enhancement == True:
            # decision_time = self.embedding_enhancement_2(x,embedding)
            # decision_time = self.orcal_EE(x, embedding, label)
            # decision_up = torch.nn.functional.interpolate(
            #     decision_time.transpose(1, 2),
            #     time_, # 501
            #     mode='linear',
            #     align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
            # return decision_time[:,:,0],decision_up,logit
            embedding = self.EE(x, embedding, label)
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        x, _ = self.detection.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.detection.fc(x)
        decision_time = torch.softmax(self.detection.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        # if enhancement:
        #     self.memory_enhancement(x_enhance,decision_time)
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2),
                time_, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0], decision_up, logit

class Join_clr(nn.Module):
    def __init__(self,model_config,inputdim,outputdim,**kwargs):
        super().__init__()
        self.encoder = Cnn14()
        self.detection = CDur_CNN14(inputdim,outputdim,**kwargs)
        self.softmax = nn.Softmax(dim=2)
        self.temperature = 5
        self.r_easy = 10 # easy 数量？
        self.r_hard = 20 # hard 数量?
        self.m = 3
        self.M = 6
        self.dropout = nn.Dropout(p=0.3)
        if model_config['pre_train']:
            self.encoder.load_state_dict(torch.load(model_config['encoder_path'])['model'])
            self.detection.load_state_dict(torch.load(model_config['CDur_path']))
    
    def get_w(self,q,k):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn/self.temperature
        attn = self.softmax(attn)
        return attn
    
    def attention_pooling(self,embeddings,mean_embedding):
        att_pool_w = self.get_w(mean_embedding,embeddings)
        embedding = torch.bmm(att_pool_w, embeddings).squeeze(1)
        # print(embedding.shape)
        # print(att_pool_w.shape)
        # print(att_pool_w[0])
        # assert 1==2
        return embedding
    
    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1) # 根据分数进行排序
        # print('_ ',_[0])
        top_k = _[:,:k]
        # print('top_k ',top_k[0])
        top_k = top_k.mean(1)
        # print('top_k ',top_k[0])
        idx_topk = idx_DESC[:, :k] # 取top_k个
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings,top_k
    
    def embedding_enhancement(self,x,embedding):
        batch, time, dim = x.shape
        mixture_embedding = self.encoder(x) # 8, 125, 128
        # print('mixture_embedding ',mixture_embedding.shape)
        # assert 1==2
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding_pre = embedding.unsqueeze(1)
        embedding_pre = embedding_pre.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding_pre), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        x, _ = self.detection.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.detection.fc(x)
        decision_time = torch.softmax(self.detection.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        selected_embeddings,top_k = self.select_topk_embeddings(decision_time[:,:,0],mixture_embedding,5)
        # print('selected_embeddings ',selected_embeddings.shape)
        # print('embedding ',embedding.shape)
        embedding_tmp = embedding.unsqueeze(1)
        #assert 1==2
        att_w = self.get_w(embedding_tmp,selected_embeddings)
        # print('att_w ',att_w[0])
        mix_embedding = torch.bmm(att_w, selected_embeddings).squeeze(1)
        # print('mix_embedding ',mix_embedding.shape)
        # assert 1==2
        att = top_k-0.55
        # print('att ',att)
        att_0 = top_k > 0.55
        # print('att_0 ',att_0)
        att = att_0 * att
        # print('att ',att.shape)
        att = att.unsqueeze(1)
        att = att.repeat(1,mix_embedding.shape[1])
        # print(att.shape)
        # print(att[0])
        # assert 1 == 2
        final_embedding = embedding + att*mix_embedding
        # print('final_embedding ',final_embedding.shape)
        # assert 1==2
        return final_embedding

    def select_topk_embeddings_clr(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1) # 根据分数进行排序
        idx_topk = idx_DESC[:, :k] # 取top_k个
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings,idx_topk

    def select_topk_embeddings_hard(self, scores, embeddings, k, hard_index):
        # print('embeddings ',embeddings.shape)
        _, idx_DESC = scores.sort(descending=True, dim=1) # 根据分数进行排序
        idx_topk = idx_DESC[:, :k] # 取top_k个
        idx_topk_tmp = idx_topk.cpu().detach().numpy()
        ans = []
        for i,idx in enumerate(idx_topk_tmp):
            tmp = np.intersect1d(idx,np.array(hard_index[i])).tolist()
            if len(tmp)==0:
                tmp = hard_index[i]
            if len(tmp) < k:
                for t in hard_index[i]:
                    tmp.append(t)
                    if len(tmp) >= k:
                        break
            if len(tmp) < k:
                for t in idx_topk_tmp[i]:
                    tmp.append(t)
                    if len(tmp) >= k:
                        break
            # print(len(tmp))
            ans.append(torch.tensor(tmp[:k]).cuda())
        idx_topk = torch.stack(ans,0).cuda()
        # print(idx_topk.shape)
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings,idx_topk
    def easy_snippets_mining(self, actionness, embeddings, k_easy):
        select_idx = torch.ones_like(actionness).cuda()
        select_idx = self.dropout(select_idx)
        actionness_drop = actionness * select_idx
        actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
        actionness_rev_drop = actionness_rev * select_idx
        easy_act,easy_act_id = self.select_topk_embeddings_clr(actionness_drop, embeddings, k_easy)
        easy_bkg,easy_bkg_id = self.select_topk_embeddings_clr(actionness_rev_drop, embeddings, k_easy)
        return easy_act,easy_act_id, easy_bkg,easy_bkg_id

    def hard_snippets_mining(self, actionness, embeddings, k_hard,labels=None):
        aness_np = actionness.cpu().detach().numpy()
        aness_median = np.median(aness_np, 1, keepdims=True)
        #aness_bin = np.where(aness_np > aness_median, 1.0, 0.0)
        aness_bin = labels.cpu().detach().numpy()
        hard_act_index, hard_bkg_index = self.hard_frame_from_label(actionness,labels)
        erosion_M = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        erosion_m = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        idx_region_inner = actionness.new_tensor(erosion_m - erosion_M)
        aness_region_inner = actionness * idx_region_inner
        hard_act,hard_act_id = self.select_topk_embeddings_hard(aness_region_inner, embeddings, k_hard,hard_act_index)

        dilation_m = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        dilation_M = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        idx_region_outer = actionness.new_tensor(dilation_M - dilation_m)
        aness_region_outer = actionness * idx_region_outer
        hard_bkg,hard_bkg_id = self.select_topk_embeddings_hard(aness_region_outer, embeddings, k_hard,hard_bkg_index)
        return hard_act,hard_act_id, hard_bkg,hard_bkg_id

    def hard_frame_from_label(self,predict,labels):
        predict_np = predict.cpu().detach().numpy()
        predict_np = predict_np > 0.5
        labels_np = labels.cpu().detach().numpy()
        labels_np = labels_np > 0.5
        change_index = []
        for i in range(predict_np.shape[0]):
            ls = np.logical_xor(predict_np[i], labels_np[i]).nonzero()[0]
            change_index.append(ls)
        hard_act_index = []
        hard_bkg_index = []
        for i,line in enumerate(change_index):
            tmp = []
            tmp2 = []
            for t in line:
                if labels_np[i,t] == True:
                    tmp.append(t)
                else:
                    tmp2.append(t)
            hard_act_index.append(tmp)
            hard_bkg_index.append(tmp2)
        return hard_act_index, hard_bkg_index
    
    def choose_sequence(self,easy_act_id,easy_bkg_id,hard_act_id,hard_bkg_id):
        easy_act_dict = {}
        easy_bkg_dict = {}
        hard_act_dict = {}
        hard_bkg_dict = {}
        k = 0
        for ea in easy_act_id:
            easy_act_dict[ea[0]] = k
            k += 1

        k = 0
        for ea in easy_bkg_id:
            easy_bkg_dict[ea[0]] = k
            k += 1
        
        k = 0
        for ea in hard_act_id:
            hard_act_dict[ea[0]] = k
            k += 1

        k = 0
        for ea in hard_bkg_id:
            hard_bkg_dict[ea[0]] = k
            k += 1

        st_easy_act_dict = sorted(easy_act_dict)
        st_easy_bkg_dict = sorted(easy_bkg_dict)
        st_hard_act_dict = sorted(hard_act_dict)
        st_hard_bkg_dict = sorted(hard_bkg_dict)
        answer_dict = {}
        for hard_ac in st_hard_act_dict:
            id = hard_ac[0]
            for easy_ac in st_easy_act_dict:
                if easy_ac[0] > id:
                    if id not in answer_dict.keys():
                        answer_dict[id] = [easy_ac[0]]
                    else:
                        answer_dict[id].append(easy_ac[0])
            

        print('easy_act_id ',easy_act_id[0])
        print('easy_bkg_id ',easy_bkg_id[0])
        print('hard_act_id ',hard_act_id[0])
        print('hard_bkg_id ',hard_bkg_id[0])
        assert 1==2
        pass

    def clr(self,predict,labels):
        predict_np = predict.cpu().detach().numpy()
        predict_np = predict_np > 0.5
        labels_np = labels.cpu().detach().numpy()
        labels_np = labels_np > 0.5
        change_indices = np.logical_xor(predict_np, labels_np).nonzero() # .nonzero()[0]


    def forward(self,x,ref,labels=None,att_pool=False,enhancement=False):
        logit = torch.zeros(1).cuda()
        embeddings  = self.encoder(ref)
        mean_embedding = embeddings.mean(1).unsqueeze(1)
        if att_pool == True:
            embedding = self.attention_pooling(embeddings,mean_embedding)
        else:
            embedding = mean_embedding.squeeze(1)
        if enhancement == True:
            embedding = self.embedding_enhancement(x,embedding)
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        x, _ = self.detection.gru(x) #  x  torch.Size([16, 125, 256])
        clr_embeddings = x
        num_segments = x.shape[1]
        k_easy = num_segments // self.r_easy
        k_hard = num_segments // self.r_hard # k_easy, k_hard 25 6
        x = self.detection.fc(x)
        decision_time = torch.softmax(self.detection.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        if labels == None:
            contrast_pairs = {
            'EA': clr_embeddings[:,0],
            'EB': clr_embeddings[:,0],
            'HA': clr_embeddings[:,0],
            'HB': clr_embeddings[:,0]}
        else:
            easy_act,easy_act_id, easy_bkg,easy_bkg_id = self.easy_snippets_mining(decision_time[:,:,0], clr_embeddings, k_easy)
            hard_act,hard_act_id, hard_bkg,hard_bkg_id = self.hard_snippets_mining(decision_time[:,:,0], clr_embeddings, k_hard,labels)
            # self.choose_sequence(easy_act_id,easy_bkg_id,hard_act_id,hard_bkg_id)
            contrast_pairs = {
                'EA': easy_act,
                'EB': easy_bkg,
                'HA': hard_act,
                'HB': hard_bkg}
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2),
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up,logit,contrast_pairs

class Join_fusion(nn.Module):
    def __init__(self,model_config,inputdim,outputdim,**kwargs):
        super().__init__()
        self.encoder = Cnn14()
        self.detection = CDur_CNN14_fusion(inputdim,outputdim,**kwargs)
        self.softmax = nn.Softmax(dim=2)
        self.temperature = 5
        if model_config['pre_train']:
            self.encoder.load_state_dict(torch.load(model_config['encoder_path'])['model'])
            self.detection.load_state_dict(torch.load(model_config['CDur_path']))
    
    def get_w(self,q,k):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn/self.temperature
        attn = self.softmax(attn)
        return attn
    
    def attention_pooling(self,embeddings,mean_embedding):
        att_pool_w = self.get_w(mean_embedding,embeddings)
        embedding = torch.bmm(att_pool_w, embeddings).squeeze(1)
        # print(embedding.shape)
        # print(att_pool_w.shape)
        # print(att_pool_w[0])
        # assert 1==2
        return embedding
    
    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1) # 根据分数进行排序
        top_k = _[:,:k]
        top_k = top_k.mean(1)
        idx_topk = idx_DESC[:, :k] # 取top_k个
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings,top_k
    
    def embedding_enhancement(self,x,embedding):
        batch, time, dim = x.shape
        mixture_embedding = self.encoder(x) # 8, 125, 128
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding_pre = embedding.unsqueeze(1)
        embedding_pre = embedding_pre.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding_pre), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        x, _ = self.detection.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.detection.fc(x)
        decision_time = torch.softmax(self.detection.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        selected_embeddings,top_k = self.select_topk_embeddings(decision_time[:,:,0],mixture_embedding,2)
        embedding_tmp = embedding.unsqueeze(1)
        att_w = self.get_w(embedding_tmp,selected_embeddings)
        mix_embedding = torch.bmm(att_w, selected_embeddings).squeeze(1)
        # print('mix_embedding ',mix_embedding.shape)
        # assert 1==2
        # att = top_k-0.55
        # # print('att ',att)
        # att_0 = top_k > 0.55
        # # print('att_0 ',att_0)
        # att = att_0 * att
        # # print('att ',att.shape)
        # att = att.unsqueeze(1)
        # att = att.repeat(1,mix_embedding.shape[1])
        # print(att.shape)
        # print(att[0])
        # assert 1 == 2
        if top_k < 0.55:
            final_embedding = embedding
        else:
            final_embedding = embedding + 0.1*mix_embedding
        return final_embedding


    def forward(self,x,ref,att_pool=False,enhancement=False):
        logit = torch.zeros(1).cuda()
        embeddings  = self.encoder(ref)
        mean_embedding = embeddings.mean(1).unsqueeze(1)
        if att_pool == True:
            embedding = self.attention_pooling(embeddings,mean_embedding)
        else:
            embedding = mean_embedding.squeeze(1)
        if enhancement == True:
            embedding = self.embedding_enhancement(x,embedding)
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        # x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        x = self.detection.fusion(embedding,x) 
        # embedding = embedding.unsqueeze(1)
        # embedding = embedding.repeat(1, x.shape[1], 1)
        # x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        x, _ = self.detection.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.detection.fc(x)
        decision_time = torch.softmax(self.detection.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2),
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up,logit
    
class AudioNet(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            Block2D(1, 64),
            Block2D(64, 64),
            nn.LPPool2d(4, (2, 2)),
            Block2D(64, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 2)),
            Block2D(128, 256),
            Block2D(256, 256),
            nn.LPPool2d(4, (1, 4)),
            Block2D(256, 512),
            Block2D(512, 512),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(512, 512, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(1024,512)
        self.outputlayer = nn.Linear(512, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # (b,512,125,1)
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        #x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7,1.) # x  torch.Size([16, 125, 369])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time,decision_up

class AudioNet2(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(128, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512,256)
        self.outputlayer = nn.Linear(256, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        # embedding = embedding.unsqueeze(1)
        # embedding = embedding.repeat(1, x.shape[1], 1)
        # x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7,1.) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time, decision_up

class LSTMCell(nn.Module):
    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory' cell.
    http://www.bioinf.jku.at/publications/older/2604.pdf
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    def forward(self, x, hidden):
        hx, cx = hidden
        x = x.view(-1, x.size(1))
        gates = self.x2h(x) + self.h2h(hx)
        #gates = gates.squeeze()
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        
        hy = torch.mul(outgate, torch.tanh(cy))
        return (hy, cy)

class LSTMCell2(nn.Module):
    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory' cell.
    http://www.bioinf.jku.at/publications/older/2604.pdf
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.x2h_f = nn.Linear(128, hidden_size, bias=bias)
        self.h2h_f = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    def forward(self, x, hidden):
        # print('x ',x.shape)
        # print('hidden ',hidden[0].shape,hidden[1].shape)
        embedding = x[:,-128:]
        # print('embdding ',embedding.shape)
        # assert 1==2
        hx, cx = hidden
        x = x.view(-1, x.size(1))
        gates = self.x2h(x) + self.h2h(hx)
        # print('gates ',gates.shape)
        # gates = gates.squeeze()
        # #if 
        # print('gates ',gates.shape)
        # assert 1==2
        ingate , cellgate, outgate = gates.chunk(3, 1)
        # print('embedding ',embedding.shape)
        # print('hx ',hx.shape)
        forgetgate = self.x2h_f(embedding) + self.h2h_f(hx)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        
        hy = torch.mul(outgate, torch.tanh(cy))
        return (hy, cy)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, bias=True):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.lstm = LSTMCell2(input_dim, hidden_dim, layer_dim) # using forget method
        self.lstm_r = LSTMCell2(input_dim, hidden_dim, layer_dim)  
        # self.fc = nn.Linear(hidden_dim, output_dim)
        self.att = Atten_1(input_dim)
    def forward(self, x):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        #print(x.shape,"x.shape")100, 28, 28
        if torch.cuda.is_available():
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
            rever_h0 =  torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
        else:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            rever_h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        # Initialize cell state
        if torch.cuda.is_available():
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
            rever_c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
        else:
            c0 = torch.zeros(self.layer_dim, x.size(0), hidden_dim)
            rever_c0 = torch.zeros(self.layer_dim, x.size(0), hidden_dim)
        outs = []
        cn = c0[0,:,:]
        hn = h0[0,:,:]
        cn_r = rever_c0[0,:,:]
        hn_r = rever_h0[0,:,:]
        for seq in range(x.size(1)):
            if seq > 2:
                input_x = x[:,seq-2:min(seq+3,x.size(1)),:]
                x_tmp = self.att(input_x)
            else:
                x_tmp = x[:,seq,:]
            reverse_seq = x.size(1)-seq-1
            if reverse_seq < x.size(1)-2:
                input_x = x[:,max(0,reverse_seq-2):reverse_seq+3,:]
                x_tmp_reverse = self.att(input_x)
            else:
                x_tmp_reverse = x[:,reverse_seq,:]
            hn, cn = self.lstm(x_tmp, (hn,cn)) 
            hn_r,cn_r = self.lstm_r(x_tmp_reverse,(hn_r,cn_r))
            # print('hn_r, hn, cn',hn_r.shape,hn.shape,cn.shape)
            # print(torch.cat((hn,hn_r),1).shape)
            # assert 1==2
            outs.append(torch.cat((hn,hn_r),1))
        outs = torch.stack(outs,1)
        # print('out ',outs.shape)
        # out = self.fc(out) 
        return outs

class LSTMModel2(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, bias=True):
        super(LSTMModel2, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.lstm = LSTMCell2(input_dim, hidden_dim, layer_dim)
        self.lstm_r = LSTMCell2(input_dim, hidden_dim, layer_dim)  
        # self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        #print(x.shape,"x.shape")100, 28, 28
        if torch.cuda.is_available():
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
            rever_h0 =  torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
        else:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            rever_h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        # Initialize cell state
        if torch.cuda.is_available():
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
            rever_c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
        else:
            c0 = torch.zeros(self.layer_dim, x.size(0), hidden_dim)
            rever_c0 = torch.zeros(self.layer_dim, x.size(0), hidden_dim)
        outs = []
        cn = c0[0,:,:]
        hn = h0[0,:,:]
        cn_r = rever_c0[0,:,:]
        hn_r = rever_h0[0,:,:]
        for seq in range(x.size(1)):
            hn, cn = self.lstm(x[:,seq,:], (hn,cn)) 
            hn_r,cn_r = self.lstm_r(x[:,x.size(1)-seq-1,:],(hn_r,cn_r))
            # print('hn_r, hn, cn',hn_r.shape,hn.shape,cn.shape)
            # print(torch.cat((hn,hn_r),1).shape)
            # assert 1==2
            outs.append(torch.cat((hn,hn_r),1))
        outs = torch.stack(outs,1)
        # print('out ',outs.shape)
        # out = self.fc(out) 
        return outs

class FilML_generator2(nn.Module):
    def __init__(self,embedding_dim,lstm_hidden_dim_q):
        super(FilML_generator2,self).__init__()
        self.lstm_q = nn.Linear(embedding_dim,lstm_hidden_dim_q)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(lstm_hidden_dim_q,2)
        # self.fc2 = nn.Linear(time_dim,4)

    def forward(self,x):
        embeddings = self.lstm_q(x)
        embeddings = self.relu(embeddings)
        # embeddings = embeddings.view(embeddings.shape[0],4,-1)
        # print('embeddings ',embeddings.shape)
        embeddings = self.fc1(embeddings)
        return embeddings

class CDur_CNN14_fiml(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = Cnn10()
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(640, 512, bidirectional=True, batch_first=True)
        self.fiml = FilML_generator2(128,128)
        # self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, outputdim)
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
    def forward(self, x, embedding): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        fi = self.fiml(embedding)
        # print('fi ',fi[0])
        # assert 1==2
        x = self.features(x,fi) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class HParams:
    def __init__(self, inputdim, outputdim, dim_neck_1,dim_neck_2, dim_enc_1,dim_enc_2,freq,
                 n_spk_emb=128):
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.dim_enc_1 = dim_enc_1
        self.dim_neck_1 = dim_neck_1
        self.dim_enc_2 = dim_enc_2
        self.dim_neck_2 = dim_neck_2
        self.freq = freq
        self.n_spk_emb = n_spk_emb

class Encoder1(nn.Module):
    def __init__(self,hparams):
        super(Encoder1, self).__init__()
        self.dim_neck_1 = hparams.dim_neck_1
        self.freq = hparams.freq
        self.dim_enc_1 = hparams.dim_enc_1
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.lstm1 = nn.LSTM(self.dim_enc_1, self.dim_neck_1, 1, batch_first=True, bidirectional=True)
        self.init_rnn_layer(self.lstm1)
    
    def init_rnn_layer(self, layer, nonlinearity='relu'):
        for name, param in layer.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
    def forward(self,x,spk_emb):
        if not hasattr(self, '_flattened'):
            self.lstm1.flatten_parameters()
        spk_emb = spk_emb.unsqueeze(1).repeat(1,x.shape[1],1)
        x = torch.cat((x,spk_emb),dim=2)
        batch, time_, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.conv_block1(x, pool_size=(2, 3), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 4), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 4), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        encoder_1_output, _ = self.lstm1(x)
        # print('encoder_1_output ',encoder_1_output.shape)
        x_forward = encoder_1_output[:, :, :self.dim_neck_1]
        x_backward = encoder_1_output[:, :, self.dim_neck_1:]
        codes_x = torch.cat((x_forward[:,self.freq-1::self.freq,:], 
                             x_backward[:,::self.freq,:]), dim=-1)
        return codes_x,encoder_1_output

class Encoder2(nn.Module):
    def __init__(self,hparams):
        super(Encoder2, self).__init__()
        self.dim_neck_2 = hparams.dim_neck_2
        self.freq = hparams.freq
        self.dim_enc_2 = hparams.dim_enc_2
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=256)
        self.conv_block3 = ConvBlock(in_channels=256, out_channels=256)
        self.lstm2 = nn.LSTM(self.dim_enc_2, self.dim_neck_2, 1, batch_first=True, bidirectional=True)
        self.init_rnn_layer(self.lstm2)
    def init_rnn_layer(self, layer, nonlinearity='relu'):
        for name, param in layer.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
    def forward(self,x):
        if not hasattr(self, '_flattened'):
            self.lstm2.flatten_parameters()
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.conv_block1(x, pool_size=(2, 3), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 4), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 4), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        encoder_2_output, _ = self.lstm2(x)
        x_forward = encoder_2_output[:, :, :self.dim_neck_2]
        x_backward = encoder_2_output[:, :, self.dim_neck_2:]
        codes_x = torch.cat((x_forward[:,self.freq-1::self.freq,:], 
                             x_backward[:,::self.freq,:]), dim=-1)
        return codes_x,encoder_2_output

class Decoder(nn.Module):
    def __init__(self,hparams):
        super(Decoder, self).__init__()
        self.dim_neck_1 = hparams.dim_neck_1
        self.dim_neck_2 = hparams.dim_neck_2
        self.dim_freq = 64
        self.decoder_lstm = nn.LSTM(self.dim_neck_1*2+self.dim_neck_2*2, 
                            256, 2, batch_first=True, bidirectional=True)
        self.linear_projection = nn.Linear(512, self.dim_freq)
        self.init_rnn_layer(self.decoder_lstm)
        self.init_fc_layer(self.linear_projection)
    
    def init_rnn_layer(self, layer, nonlinearity='relu'):
        for name, param in layer.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    def init_fc_layer(self, layer, nonlinearity='relu'):
        """Initialize a Linear or Convolutional layer. """
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0.)

    def forward(self,feature_target,feature_others,time_scale):
        if not hasattr(self.decoder_lstm, '_flattened'):
            self.decoder_lstm.flatten_parameters()
            setattr(self.decoder_lstm, '_flattened', True)
        x = torch.cat((feature_target,feature_others),2)
        x,_ = self.decoder_lstm(x)
        decoder_output = self.linear_projection(x)
        #print('decoder_output ',decoder_output.shape)
        # print('time_ ',time_)
        decoder_output_up = torch.nn.functional.interpolate(
                decoder_output.transpose(1, 2), # [16, 2, 125]
                time_scale, mode='linear', align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decoder_output_up

class Detection2(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.features = Cnn10()
        self.inputdim = hparams.inputdim
        self.outputdim = hparams.outputdim
        self.dim_neck_1 = hparams.dim_neck_1
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,self.inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(512+2*self.dim_neck_1, 512, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, self.outputdim)
        self.outputlayer.apply(init_weights)
    
    def forward(self, x, embedding,time_scale=None): 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # embedding = embedding.unsqueeze(1)
        # embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding[:,:time_scale,:]), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class Detection(nn.Module):
    def __init__(self,hparams):
        super(Detection, self).__init__()
        inputdim = hparams.inputdim
        outputdim = hparams.outputdim
        self.dim_neck_1 = hparams.dim_neck_1
        self.n_spk_emb = hparams.n_spk_emb
        self.gru = nn.GRU(2*self.dim_neck_1+self.n_spk_emb, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512,256)
        self.outputlayer = nn.Linear(256, outputdim)
        self.outputlayer.apply(init_weights)
        self.init_rnn_layer(self.gru)
    
    def init_rnn_layer(self, layer, nonlinearity='relu'):
        for name, param in layer.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
    
    def init_fc_layer(self, layer, nonlinearity='relu'):
        """Initialize a Linear or Convolutional layer. """
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0.)

    def forward(self,x,embedding,time_scale):
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time_scale, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class DCnet(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super(DCnet, self).__init__()
        dim_neck_1 = 16
        dim_neck_2 = 24
        dim_enc_1 =  1024
        dim_enc_2 =  256
        self.freq = 5
        self.dim_freq = 64
        hparams = HParams(inputdim, outputdim,dim_neck_1,dim_neck_2,dim_enc_1,dim_enc_2,self.freq)
        self.encoder_target = Encoder1(hparams)
        self.encoder_others = Encoder2(hparams)
        self.decoder = Decoder(hparams)
        self.detection = Detection2(hparams)
    
    def forward(self,input_spec,embedding):
        shape_input = input_spec.shape
        target_codes,encoder_outputs1 = self.encoder_target(input_spec,embedding)
        #print('target_codes,encoder_outputs1',target_codes.shape,encoder_outputs1.shape)
        others_codes,encoder_outputs2 = self.encoder_others(input_spec)
        up_target_codes = target_codes.repeat_interleave(self.freq, dim=1)
        up_others_codes = others_codes.repeat_interleave(self.freq, dim=1)
        # calculate similar loss
        # decoder
        recon_spec = self.decoder(up_target_codes,up_others_codes,shape_input[1])
        #print('recon_spec ',recon_spec.shape)
        # extraction
        decision_time,decision_up = self.detection(input_spec,up_target_codes,shape_input[1])
        return decision_time, decision_up, recon_spec

class DCnet2(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super(DCnet2, self).__init__()
        dim_neck_1 = 32
        dim_neck_2 = 32
        dim_enc_1 =  1024
        dim_enc_2 =  256
        self.freq = 5
        self.dim_freq = 64
        hparams = HParams(inputdim, outputdim,dim_neck_1,dim_neck_2,dim_enc_1,dim_enc_2,self.freq)
        self.encoder_target = Encoder1(hparams)
        self.encoder_others = Encoder2(hparams)
        self.decoder = Decoder(hparams)
        self.detection = Detection2(hparams)
    
    def forward(self,input_spec,embedding):
        shape_input = input_spec.shape
        target_codes,encoder_outputs1 = self.encoder_target(input_spec,embedding)
        #print('target_codes,encoder_outputs1',target_codes.shape,encoder_outputs1.shape)
        others_codes,encoder_outputs2 = self.encoder_others(input_spec)
        up_target_codes = target_codes.repeat_interleave(self.freq, dim=1)
        up_others_codes = others_codes.repeat_interleave(self.freq, dim=1)
        # calculate similar loss
        # decoder
        recon_spec = self.decoder(up_target_codes,up_others_codes,shape_input[1])
        #print('recon_spec ',recon_spec.shape)
        # extraction
        decision_time,decision_up = self.detection(input_spec,up_target_codes,shape_input[1])
        decision_time2,decision_up2 = self.detection(input_spec,up_others_codes,shape_input[1])
        return decision_time, decision_up, decision_time2, decision_up2,recon_spec
# class EventInference(nn.Module):
#     def __init__(self):
#         super(EventInference, self).__init__()

####################################
# The following Transformer modules are modified from Yu-Hsiang Huang's code:
# https://github.com/jadore801120/attention-is-all-you-need-pytorch
# class ScaledDotProductAttention(nn.Module):
#     """Scaled Dot-Product Attention"""

#     def __init__(self, temperature, attn_dropout=0.1):
#         super().__init__()
#         self.temperature = temperature
#         self.dropout = nn.Dropout(attn_dropout)
#         self.softmax = nn.Softmax(dim=2)

#     def forward(self, q, k, v, mask=None):

#         attn = torch.bmm(q, k.transpose(1, 2))
#         attn = attn / self.temperature

#         if mask is not None:
#             attn = attn.masked_fill(mask, -np.inf)

#         attn = self.softmax(attn)
#         attn = self.dropout(attn)
#         output = torch.bmm(attn, v)

#         return output, attn


# class MultiHead(nn.Module):
#     """Multi-Head Attention module."""

#     def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
#         super().__init__()

#         self.n_head = n_head
#         self.d_k = d_k
#         self.d_v = d_v

#         self.w_qs = nn.Linear(d_model, n_head * d_k)
#         self.w_ks = nn.Linear(d_model, n_head * d_k)
#         self.w_vs = nn.Linear(d_model, n_head * d_v)
#         nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
#         nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
#         nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
#         self.w_qs.bias.data.fill_(0)
#         self.w_ks.bias.data.fill_(0)
#         self.w_vs.bias.data.fill_(0)

#         self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
#         self.layer_norm = nn.LayerNorm(d_model)

#         self.fc = nn.Linear(n_head * d_v, d_model)
#         nn.init.xavier_normal_(self.fc.weight)
#         self.fc.bias.data.fill_(0)

#         self.dropout = nn.Dropout(dropout)


#     def forward(self, q, k, v, mask=None):

#         d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

#         sz_b, len_q, _ = q.size()   # (batch_size, 80, 512)
#         sz_b, len_k, _ = k.size()
#         sz_b, len_v, _ = v.size()

#         residual = q

#         q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) # (batch_size, T, 8, 64)
#         k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
#         v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

#         q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk, (batch_size*8, T, 64)
#         k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
#         v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

#         # mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
#         output, attn = self.attention(q, k, v, mask=mask)   # (n_head * batch_size, T, 64), (n_head * batch_size, T, T)
        
#         output = output.view(n_head, sz_b, len_q, d_v)  # (n_head, batch_size, T, 64)
#         output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv), (batch_size, T, 512)
#         output = F.relu_(self.dropout(self.fc(output)))
#         return output
# class TimeShift(nn.Module):
#     def __init__(self, mean, std):
#         super().__init__()
#         self.mean = mean
#         self.std = std

#     def forward(self, x):
#         if self.training:
#             shift = torch.empty(1).normal_(self.mean, self.std).int().item()
#             x = torch.roll(x, shift, dims=2)
#         return x

# class OverlapPatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """

#     def __init__(self, tdim, fdim, patch_size=7, stride=4, in_chans=3, embed_dim=768):
#         super().__init__()
#         img_size = (tdim, fdim)
#         patch_size = to_2tuple(patch_size)

#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.H, self.W = img_size[0] // stride, img_size[1] // stride
#         self.num_patches = self.H * self.W
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
#                               padding=(patch_size[0] // 3, patch_size[1] // 3))
#         self.norm = nn.LayerNorm(embed_dim)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x):
#         x = self.proj(x)
#         _, _, H, W = x.shape
#         x = x.flatten(2).transpose(1, 2)
#         x = self.norm(x)

#         return x, H, W

# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.dwconv = DWConv(hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#         self.linear = linear
#         if self.linear:
#             self.relu = nn.ReLU()
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x, H, W):
#         x = self.fc1(x)
#         if self.linear:
#             x = self.relu(x)
#         x = self.dwconv(x, H, W)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

# class DWConv(nn.Module):
#     def __init__(self, dim=768):
#         super(DWConv, self).__init__()
#         self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         x = x.transpose(1, 2).view(B, C, H, W)
#         x = self.dwconv(x)
#         x = x.flatten(2).transpose(1, 2)

#         return x

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         self.linear = linear
#         self.sr_ratio = sr_ratio
#         if not linear:
#             if sr_ratio > 1:
#                 self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
#                 self.norm = nn.LayerNorm(dim)
#         else:
#             self.pool = nn.AdaptiveAvgPool2d(7)
#             self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
#             self.norm = nn.LayerNorm(dim)
#             self.act = nn.GELU()
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

#         if not self.linear:
#             if self.sr_ratio > 1:
#                 x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
#                 x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
#                 x_ = self.norm(x_)
#                 kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#             else:
#                 kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         else:
#             x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
#             x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
#             x_ = self.norm(x_)
#             x_ = self.act(x_)
#             kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x


# class Pooling(nn.Module):
#     """
#     Implementation of pooling for PoolFormer
#     --pool_size: pooling size
#     """
#     def __init__(self, pool_size=3):
#         super().__init__()
#         self.pool = nn.AvgPool2d(
#             pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

#     def forward(self, x):
#         return self.pool(x) - x

# class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        #self.norm3 = norm_layer(dim)
        #self.token_mixer = Pooling(pool_size=3)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)
        #layer_scale_init_value=1e-5
        #self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        #self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        #x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x), H, W))
        #x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x), H, W))
        #x = x + self.drop_path(self.token_mixer(self.norm3(x)))
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


# class PVT_Transformer_shift2(nn.Module):
#     def __init__(self, inputdim, outputdim,time_resolution,**kwargs):
#         super(PVT_Transformer_shift2, self).__init__()
#         window = 'hann'
#         center = True
#         pad_mode = 'reflect'
#         ref = 1.0
#         amin = 1e-10
#         top_db = None
#         self.pvt_transformer = PyramidVisionTransformerV2(tdim=1001,
#                                 fdim=64,
#                                 patch_size=6,
#                                 stride=4,
#                                 in_chans=1,
#                                 num_classes=2,
#                                 embed_dims=[64, 128],
#                                 depths=[3, 4],
#                                 num_heads=[1, 2],
#                                 mlp_ratios=[8, 8],
#                                 qkv_bias=True,
#                                 qk_scale=None,
#                                 drop_rate=0.0,
#                                 drop_path_rate=0.1,
#                                 sr_ratios=[8, 4],
#                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                                 pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth'
#                                 )
#         n_head = 8
#         n_hid = 512
#         d_k = 64
#         d_v = 64
#         dropout = 0.2
#         self.fc2 = nn.Linear(128, 512, bias=True)
#         # self.multihead = MultiHead(8, 640, 64, 64, dropout)
#         self.temp_pool = LinearSoftPool()
#         self.fc = nn.Linear(640, 2, bias=True)
#         self.init_weights()

#     def init_weights(self):
#         init_layer(self.fc)

#     def forward(self, input, embedding, one_hot=None):
#         """Input: (batch_size, times_steps, freq_bins)"""
#         batch, time, dim = input.shape
#         x = input.unsqueeze(1) # (b,1,t,d) 
#         # print('x0 ',x.shape)
#         # #print(x.shape)   #torch.Size([10, 1, 1001, 64])
#         x = self.pvt_transformer(x) # 
#         # print('x ',x.shape)
#         x = torch.mean(x, dim=3)
#         x = x.transpose(1, 2).contiguous() # [2, 125, 128]
#         x = self.fc2(x) # [2, 125, 512]
#         embedding = embedding.unsqueeze(1)
#         embedding = embedding.repeat(1, x.shape[1], 1)
#         x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim] 2, 125, 640]
#         # x = self.multihead(x, x, x)
#         # embedding = x.transpose(1, 2).contiguous()  # (batch_size, feature_maps, time_steps)
#         # Framewise output
#         #x = x.transpose(1, 2).contiguous()
#         decision_time = torch.softmax(self.fc(x), dim=2)
#         clipwise_output = self.temp_pool(x, decision_time).clamp(1e-7, 1.).squeeze(1)
#         #print(framewise_output.shape)    #torch.Size([10, 100, 17])
#         decision_up = torch.nn.functional.interpolate(
#                 decision_time.transpose(1, 2), # [16, 2, 125]
#                 time, # 501
#                 mode='linear',
#                 align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
#         # decision_time_up = interpolate(decision_time, interpolate_ratio)   
#         # print('decision_up ',decision_up.shape)
#         # assert 1==2 
#         return decision_time[:,:,0],decision_up


# class PyramidVisionTransformerV2(nn.Module):
    def __init__(self, tdim=1001, fdim=64, patch_size=16, stride=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=2, linear=False, pretrained=None):
        super().__init__()
        # self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.linear = linear
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(tdim=tdim if i == 0 else tdim // (2 ** (i + 1)),
                                            fdim=fdim if i == 0 else tdim // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=stride if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])
            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        #self.n = nn.Linear(125, 250, bias=True)
        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        self.init_weights(pretrained)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            #print(x.shape)
            for blk in block:
                x = blk(x, H, W)
            #print(x.shape)
            x = norm(x)
            #if i != self.num_stages - 1:
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            '''
            if i == 1 or i == 2:
                #x = interpolate_median(x, 4)
                #x = torch.nn.functional.interpolate(x, (x.shape[2]*2, x.shape[3]), mode='bilinear', align_corners=False)
                x = self.n(x.transpose(2, 3))
                x = x.transpose(2, 3)
                #print(x.shape)
            '''
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x