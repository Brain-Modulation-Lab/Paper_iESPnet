#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create the PyTorch NN model class. 
This is inspired in this post
https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch
@author: vpeterson
"""

import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
#%%
#trying to speed up training
# 1. adding bias=false (sin layer normalization is done)
GROUPS = 1 # if group is equal to input channels, then deepwise conv2D
NFREQ = 120
NTIME = 181
MOMENTUM = 0.01
class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2, groups=GROUPS, bias=False)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2, groups=GROUPS, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = CNNLayerNorm(n_feats)
        self.norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)
    
    

class ResidualCNNbatch(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNNbatch, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2, groups=GROUPS, bias=False)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2, groups=GROUPS, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm2d(in_channels, track_running_stats=True, momentum=MOMENTUM)
        self.norm2 = nn.BatchNorm2d(out_channels, track_running_stats=True, momentum=MOMENTUM)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)


class GRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout,  batch_first=True):
        super(GRU, self).__init__()

        self.GRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.GRU(x)
        x = self.dropout(x)
        return x

class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.relu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x

#%% MODEL DEF 
class iESPnet(nn.Module):
    # this net with use timeConv and FreqConv
    # then concatenate the output and do a trad 2D conv
    
    def FreqConv(self, in_channels, out_channels, n_freq=NFREQ, n_time=NTIME):
        block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, (n_freq, 1), stride=(1,1), padding=(n_freq-1, 0), dilation=(2,1), groups=GROUPS, bias=False),
                torch.nn.ReLU(),
                torch.nn.InstanceNorm2d(out_channels, track_running_stats=True, affine=True, momentum=MOMENTUM)
                )
        return  block
    
    
    def TimeConv(self, in_channels, out_channels, n_freq=NFREQ, n_time=NTIME):
        block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, (1, n_time), stride=(1,1), padding=(0, n_time-1), dilation=(1,2), groups=GROUPS, bias=False),
                torch.nn.ReLU(),
                torch.nn.InstanceNorm2d(out_channels, track_running_stats=True, affine=True, momentum=MOMENTUM)
                )
        return  block
    
    
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, out_ch, dropout, n_freq=NFREQ, n_time=NTIME):
        super(iESPnet, self).__init__()
        # freqConv
        self.freqcnn = self.FreqConv(4, out_ch[0])
        # TimeConv
        self.timecnn = self.TimeConv(4, out_ch[1])      
        # cnn for extracting heirachal features. groups=in_ch
        self.cnn_ori = nn.Conv2d(4, out_ch[2], 3, stride=(2,1), padding=1, groups=GROUPS, bias=False)
        self.cnn = nn.Conv2d(sum(out_ch[:2]), out_ch[2], 3, stride=(2,1), padding=1, groups=GROUPS, bias=False)

        # n residual cnn layers
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNNbatch(out_ch[2], out_ch[2], kernel=3, stride=1, dropout=dropout, n_feats=n_freq) 
            for _ in range(n_cnn_layers)
        ])
        self.rescnn_layers_ori = nn.Sequential(*[
            ResidualCNNbatch(out_ch[2], out_ch[2], kernel=3, stride=1, dropout=dropout, n_feats=n_freq) 
            for _ in range(n_cnn_layers)
        ])

        # avpooling
        self.avgpool = nn.AdaptiveAvgPool2d((n_freq//3, n_time))
       
        self.rnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=(n_freq//3)*(2*out_ch[2]) if i==0 else 2*rnn_dim[i-1],
                             hidden_size=rnn_dim[i], dropout=dropout, batch_first=i==0)
             for i in range(n_rnn_layers)
            ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim[-1]*2, rnn_dim[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim[-1], n_class)
        )


    def forward(self, x):
        x0 = x
        x1 = self.freqcnn(x0)
        x2 = self.timecnn(x0)
        x12 = torch.cat((x1, x2), dim=1)

        x0 = self.cnn_ori(x0)
        x12 = self.cnn(x12)
        x3 = self.rescnn_layers_ori(x0)
        x4 = self.rescnn_layers(x12)
        
        x = torch.cat((x3, x4), dim=1)
        
        x = self.avgpool(x)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        
        x = self.rnn_layers(x)

        x = self.classifier(x)

        return  torch.squeeze(x)

