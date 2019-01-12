#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 01:11:23 2019

@author: bharat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN_Text(nn.Module):
    
    def __init__(self, embed_num, embed_dim, class_num, kernel_num, static):
        super(CNN_Text, self).__init__()
        self.embed_num = embed_num
        self.embed_dim =embed_dim
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.static =static
        
        V = self.embed_num
        D = self.embed_dim
        C = self.class_num
        dropout = 0.5
        Ci = 1
        Co = self.kernel_num# number of filters
       

        self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        
        """Parameters - kernel sizes"""
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(3*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        
        if self.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
    
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit
