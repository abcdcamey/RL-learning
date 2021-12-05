# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2021/12/1 8:19 下午
# Description : 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 128):
        """
        多层感知机，全连接网络
        :param input_dim:
        :param output_dim:
        :param hidden_dim:
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
