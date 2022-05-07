# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2022/5/1 8:27 下午
# Description : 
"""
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
class ActorCritic(nn.Module):
    '''
    A2C模型：包含Actor和Critic
    '''
    def __init__(self, input_dim, output_dim, hidden_dim):
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        value = self.critic(x)
        dist = self.actor(x)
        dist = Categorical(dist)
        return dist, value




class A2C:

    def __init__(self, state_dim, action_dim, cfg) -> None:
        self.gamma = cfg.gamma
        self.device = cfg.device
        self.model = ActorCritic(state_dim, action_dim, cfg.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())

    def compute_returns(self, next_value, rewards, masks):
        pass