# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2021/11/30 6:11 下午
# Description : 
"""
import numpy as np
from collections import defaultdict
import torch
import dill
class Sarsa(object):
    def __init__(self, cfg):
        self.action_dim = cfg.action_dim # number of actions
        self.lr = cfg.lr
        self.gamma = cfg.gamma # reward 衰减系数
        self.epsilon = cfg.epsilon #epsilon策略选择
        self.Q_table = defaultdict(lambda: np.zeros(self.action_dim))

    def choose_action(self, state):
        best_action = np.argmax(self.Q_table[state])
        action_probs = np.ones(self.action_dim, dtype=float) * self.epsilon / self.action_dim
        action_probs[best_action] += (1.0-self.epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def update(self, state, action, reward, next_state, next_action, done):
        Q_predict = self.Q_table[state][action]
        if done:
            Q_target = reward
        else:
            Q_target = reward + self.gamma * self.Q_table[next_state][next_action]
        self.Q_table[state][action] += self.lr * (Q_target - Q_predict)

    def save(self,path):
        '''把 Q表格 的数据保存到文件中
                '''
        torch.save(
            obj=self.Q_table,
            f=path + "sarsa_model.pkl",
            pickle_module=dill
        )
    def load(self, path):
        '''从文件中读取数据到 Q表格
        '''
        self.Q_table =torch.load(f=path+'sarsa_model.pkl',pickle_module=dill)