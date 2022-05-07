# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2022/5/6 10:38 下午
# Description : 
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from TD3.memory import ReplayBuffer

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = action_dim

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return self.max_action * torch.tanh(self.linear3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        #Q1
        self.linear1 = nn.Linear(state_dim+action_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        #Q2
        self.linear4 = nn.Linear(state_dim+action_dim, 256)
        self.linear5 = nn.Linear(256, 256)
        self.linear6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.linear1(sa))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)

        q2 = F.relu(self.linear4(sa))
        q2 = F.relu(self.linear5(q2))
        q2 = self.linear6(q2)

        return q1, q2


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, cfg):
        self.max_action = max_action
        self.gamma = cfg.gamma
        self.lr = cfg.lr
        self.policy_noise = cfg.policy_noise
        self.noise_clip = cfg.noise_clip
        self.policy_freq = cfg.policy_freq
        self.batch_size = cfg.batch_size
        self.device = cfg.device
        self.total_it = 0

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.memory = ReplayBuffer(state_dim, action_dim)

    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.action(state).cpu().data.numpy().flatten()

    def update(self):
        self.total_it += 1
        state, action, next_state, reward, done = self.memory.sample(self.batch_size)

        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + done*(self.gamma * target_Q)

        current_Q1, current_Q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:

            actor_loss = - self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.lr * param.data + (1 - self.lr) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.lr * param.data + (1 - self.lr) * target_param.data)





    def save(self, path):
        torch.save(self.critic.state_dict(), path + "td3_critic")
        torch.save(self.critic_optimizer.state_dict(), path + "td3_critic_optimizer")

        torch.save(self.actor.state_dict(), path + "td3_actor")
        torch.save(self.actor_optimizer.state_dict(), path + "td3_actor_optimizer")

    def load(self, path):
        self.critic.load_state_dict(torch.load(path + "td3_critic"))
        self.critic_optimizer.load_state_dict(torch.load(path + "td3_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(path + "td3_actor"))
        self.actor_optimizer.load_state_dict(torch.load(path + "td3_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
