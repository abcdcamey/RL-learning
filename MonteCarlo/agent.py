# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2021/11/15 6:00 下午
# Description : 
"""
from collections import defaultdict
import numpy as np
import torch
import dill
class FirstVisitMC:

    def __init__(self, cfg):
        self.action_dim = cfg.action_dim #action的维度
        self.epsilon = cfg.epsilon #
        self.gamma = cfg.gamma # 奖励衰减参数
        self.Q_table = defaultdict(lambda: np.zeros(self.action_dim)) #Q table 存储参数,默认key的value是维度为action_dim的list
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)

    def choose_action(self, state):
        """
        如果Qtable里面有历史state，则找到历史最优action，然后使用epsilon参数加大该action的概率
        如果Qtable里面没有历史state，则随机选择一个action
        """
        if state in self.Q_table.keys():
            best_action = np.argmax(self.Q_table[state])
            action_probs = np.ones(self.action_dim, dtype=float) * self.epsilon / self.action_dim
            action_probs[best_action] += (1.0 - self.epsilon)
            action = np.random.choice(np.arange(0, self.action_dim), p=action_probs)
        else:
            action = np.random.randint(0, self.action_dim)
        return action

    def save(self, path):
        '''保存Q_tabel'''
        torch.save(
            obj=self.Q_table,
            f=path+"Q_table",
            pickle_module=dill
        )

    def load(self, path):
        '''读取Q_table'''
        self.Q_table = torch.load(f=path+"Q_table")

    def update(self, one_ep_transition):
        """
        在一次episode的state,action集合中，找到第一次出现的下标，然后计算从该时刻开始到episode结束时对应的平均return,更新Q_table
        """
        sa_in_episode = set([(tuple(x[0]), x[1]) for x in one_ep_transition])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            first_occurence_idx = next(i for i, x in enumerate(one_ep_transition)
                                   if x[0]==state and x[1]==action)

            G = sum([x[2]*(self.gamma**i) for i, x in enumerate(one_ep_transition[first_occurence_idx:])])
            self.returns_sum[sa_pair]+=G
            self.returns_count[sa_pair]+=1.0
            self.Q_table[state][action] = self.returns_sum[sa_pair] / self.returns_count[sa_pair]

