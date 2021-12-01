# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2021/11/30 8:05 下午
# Description : 
"""
import datetime
import sys, os
import torch
import gym
from ENVS.racetrack_env import RacetrackEnv

from Sarsa.agent import Sarsa
from COMMON.utils import save_results, make_dir
from COMMON.plot import plot_rewards, plot_rewards_cn


curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前路径
parent_path = os.path.dirname(curr_path) # 父路径，这里就是我们的项目路径
sys.path.append(parent_path)

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time

class SarsaConfig:
    def __init__(self):
        self.algo = "Sarsa"
        self.env = "Racetrack"
        self.result_path = curr_path + "/outputs/" + self.env + '/' + curr_time + '/results/'  # path to save results
        self.model_path = curr_path + "/outputs/" + self.env + '/' + curr_time + '/models/'  # path to save models
        self.train_eps = 200
        self.eval_eps = 50
        self.epsilon = 0.15
        self.gamma = 0.9
        self.lr = 0.2
        self.n_steps = 2000
        self.action_dim = 9

def env_agent_config(cfg):
    env = RacetrackEnv()
    agent = Sarsa(cfg)
    return env, agent

def train(cfg, env, agent):
    print("training...")
    rewards = []
    ma_rewards = []
    for i_episode in range(cfg.train_eps):
        print(i_episode)
        state = env.reset()
        ep_reward = 0
        while True:

            action = agent.choose_action(state)
            #print(action)
            next_state, reward, done = env.step(action)
            ep_reward += reward
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)
            state = next_state
            if done:
                break
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        rewards.append(ep_reward)
        if (i_episode + 1) % 10 == 0:
            print("Episode:{}/{}: Reward:{}".format(i_episode + 1, cfg.train_eps, ep_reward))
    return rewards, ma_rewards

def eval(cfg, env, agent):
    rewards = []
    ma_rewards = []
    for i_episode in range(cfg.eval_eps):
        state = env.reset()
        ep_reward = 0
        while True:
            #env.render()
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            ep_reward+=reward
            state = next_state
            if done:
                break
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        rewards.append(ep_reward)
        if (i_episode+1)%10==0:
            print("Episode:{}/{}: Reward:{}".format(i_episode+1, cfg.eval_eps,ep_reward))
    print('Complete evaling！')
    return rewards, ma_rewards

if __name__=="__main__":
    cfg = SarsaConfig()
    env, agent = env_agent_config(cfg)
    rewards,ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, tag="train", env=cfg.env, algo=cfg.algo, path=cfg.result_path)

    env, agent = env_agent_config(cfg)
    agent.load(path=cfg.model_path)
    rewards, ma_rewards = eval(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='eval', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, tag="eval", env=cfg.env, algo=cfg.algo, path=cfg.result_path)