# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2021/11/17 1:45 下午
# Description : 
"""
from collections import defaultdict
import numpy as np
import math
import dill
import torch
class QLearning(object):
    def __init__(self, cfg):
        self.action_dim = cfg.action_dim
        self.lr = cfg.lr #learning rate
        self.gamma = cfg.gamma # reward的衰减系数
        self.epsilon = 0 # 选择action的参数
        self.sample_count = 0
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table = defaultdict(lambda: np.zeros(self.action_dim))

    def choose_action(self, state):
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) # epsilon 指数递减

        # e-greedy 策略
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)])
        else:
            action = np.random.choice(self.action_dim)
        return action

    def predict(self, state):
        action = np.argmax(self.Q_table[str(state)])
        return action

    def update(self, state, action, reward, next_state, done):
        Q_predict = self.Q_table[str(state)][action]
        if done:
            Q_target = reward
        else:
            Q_target = reward + self.gamma * np.max(self.Q_table[str(next_state)])
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict)

    def save(self, path):
        torch.save(
            obj=self.Q_table,
            f=path + "Qleaning_model.pkl",
            pickle_module=dill
        )
        print("保存模型成功！")

    def load(self, path):
        self.Q_table = torch.load(f=path + 'Qleaning_model.pkl', pickle_module=dill)
        print("加载模型成功！")