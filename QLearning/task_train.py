# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2021/11/18 11:48 上午
# Description : 
"""
import datetime
import sys, os
import torch
import gym
from ENVS.gridworld_env import CliffWalkingWapper
from QLearning.agent import QLearning
from COMMON.utils import save_results, make_dir
from COMMON.plot import plot_rewards, plot_rewards_cn

curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前路径
parent_path = os.path.dirname(curr_path) # 父路径，这里就是我们的项目路径
sys.path.append(parent_path)


curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 获取当前时间

class QlearningConfig:
    '''训练相关参数'''
    def __init__(self):
        self.algo = "Q-learning"
        self.env = "CliffWalking-v0"
        self.result_path = curr_path+"/outputs/" +self.env+'/'+curr_time+'/results/'
        self.model_path = curr_path+"/outputs/" +self.env+'/'+curr_time+'/models/'  # 保存模型的路径
        self.train_eps = 400
        self.eval_eps = 30
        self.gamma = 0.9
        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.epsilon_decay = 300
        self.lr = 0.1
        self.action_dim = 0
        self.states_dim = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测GPU

def env_agent_config(cfg, seed = 1):
    env = gym.make(cfg.env)
    env = CliffWalkingWapper(env)
    env.seed(seed)
    cfg.states_dim = env.observation_space.n
    cfg.action_dim = env.action_space.n
    agent = QLearning(cfg)
    return env, agent

def train(cfg, env, agent):
    print('开始训练！')
    print(f'环境:{cfg.env}, 算法:{cfg.algo}, 设备:{cfg.device}')
    rewards = []
    ma_rewards = []
    for i_ep in range(cfg.train_eps):
        ep_reward = 0
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done , _ = env.step(action)
            print(reward)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
    print("回合数：{}/{}，奖励{:.1f}".format(i_ep + 1, cfg.train_eps, ep_reward))
    print('完成训练！')
    return rewards, ma_rewards

def eval(cfg,env,agent):
    print('开始测试！')

    rewards = []  # 记录所有回合的奖励
    ma_rewards = [] # 滑动平均的奖励
    for i_ep in range(cfg.eval_eps):
        ep_reward = 0  # 记录每个episode的reward
        state = env.reset()  # 重置环境, 重新开一局（即开始新的一个回合）
        while True:
            env.render()
            action = agent.predict(state)  # 根据算法选择一个动作
            next_state, reward, done, _ = env.step(action)  # 与环境进行一个交互
            state = next_state  # 更新状态
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f"回合数：{i_ep+1}/{cfg.eval_eps}, 奖励：{ep_reward:.1f}")
    print('完成测试！')
    return rewards,ma_rewards

if __name__ == "__main__":
    cfg = QlearningConfig()

    #训练
    env, agent = env_agent_config(cfg, seed=0)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)  # 创建文件夹
    agent.save(path=cfg.model_path)  # 保存模型

    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)  # 保存结果
    plot_rewards_cn(rewards, ma_rewards, tag="train", env=cfg.env, algo=cfg.algo, path=cfg.result_path)

    # 验证
    env, agent = env_agent_config(cfg, seed=10)
    agent.load(path=cfg.model_path)  # 加载模型
    rewards, ma_rewards = eval(cfg, env, agent)

    save_results(rewards, ma_rewards, tag='eval', path=cfg.result_path)
    plot_rewards_cn(rewards, ma_rewards, tag="eval", env=cfg.env, algo=cfg.algo, path=cfg.result_path)
