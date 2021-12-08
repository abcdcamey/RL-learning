# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2021/12/6 2:17 下午
# Description : 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np
from COMMON.memory import ReplayBuffer
from COMMON.model import MLP
class DoubleDQN:
    def __init__(self, cfg, state_dim, action_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = cfg.device
        self.gamma = cfg.gamma
        # e-greedy 策略相关参数
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: cfg.epsilon_end+(cfg.epsilon_start-cfg.epsilon_end)*math.exp(-1*self.frame_idx/ cfg.epsilon_decay)
        self.batch_size = cfg.batch_size
        self.policy_net = MLP(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = MLP(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)

        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr = cfg.lr)
        self.loss = 0
        self.memory = ReplayBuffer(cfg.memory_capacity)

    def predict(self, state):
        with torch.no_grad():
            state = torch.tensor(
                [state], device=self.device, dtype=torch.float32
            )
            q_value = self.policy_net(state)
            # tensor.max(1)返回每行的最大值以及对应的下标
            # 所以tensor.max(1)[1]返回最大值对应的下标，即action
            action = q_value.max(1)[1].item()
        return action

    def choose_action(self, state):
        self.frame_idx+=1
        if random.random()>self.epsilon(self.frame_idx):
            action = self.predict(state)
        else:
            action = random.randrange(self.action_dim)
        return action

    def update(self):

        if len(self.memory) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size
        )
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)
        q_values = self.policy_net(state_batch)
        next_q_values = self.policy_net(next_state_batch)
        q_value = q_values.gather(dim=1, index=action_batch)

        # Double DQN
        next_target_values = self.target_net(next_state_batch)
        next_target_q_value = next_target_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        q_target = reward_batch + self.gamma * next_target_q_value * (1 - done_batch)

        # Nature DQN
        # next_target_values = self.target_net(next_state_batch)
        # next_target_q_value = next_target_values.max(1)[0].detach()
        # q_target = reward_batch + self.gamma * next_target_q_value * (1 - done_batch)
        self.loss = nn.MSELoss()(q_value, q_target.unsqueeze(1))

        self.optimizer.zero_grad()
        self.loss.backward()
        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()  # 更新模型

    def save(self, path):
        torch.save(self.target_net.state_dict(), path+'checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path+'checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)

