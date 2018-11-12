# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 09:03:43 2018

@author: Christopher
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

class AttributeNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size,output_size):
        super(AttributeNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size,output_size):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
