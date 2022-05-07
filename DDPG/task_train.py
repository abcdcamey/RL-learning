# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2022/5/6 3:23 下午
# Description : 
"""
import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
sys.path.append(parent_path) # 添加路径到系统路径sys.path
from DDPG.env import OUNoise

import datetime
import gym
import torch

from DDPG.env import NormalizedActions
from DDPG.agent import DDPG
from COMMON.utils import save_results,make_dir
from COMMON.utils import plot_rewards

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
algo_name = 'DDPG'  # 算法名称
env_name = 'Pendulum-v0'  # 环境名称，gym新版本（约0.21.0之后）中Pendulum-v0改为Pendulum-v1


class DDPGConfig:
    def __init__(self):
        self.algo_name = algo_name # 算法名称
        self.env_name = env_name # 环境名称
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测GPU
        self.train_eps = 300 # 训练的回合数
        self.test_eps = 50 # 测试的回合数
        self.gamma = 0.99 # 折扣因子
        self.critic_lr = 1e-3 # 评论家网络的学习率
        self.actor_lr = 1e-4 # 演员网络的学习率
        self.memory_capacity = 8000 # 经验回放的容量
        self.batch_size = 128 # mini-batch SGD中的批量大小
        self.target_update = 2 # 目标网络的更新频率
        self.hidden_dim = 256 # 网络隐藏层维度
        self.soft_tau = 1e-2 # 软更新参数

class PlotConfig:
    def __init__(self) -> None:
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name # 环境名称
        self.result_path = curr_path+"/outputs/" + self.env_name + \
            '/'+curr_time+'/results/'  # 保存结果的路径
        self.model_path = curr_path+"/outputs/" + self.env_name + \
            '/'+curr_time+'/models/'  # 保存模型的路径
        self.save = True # 是否保存图片
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU

def env_agent_config(cfg, seed=1):
    env = NormalizedActions(gym.make(cfg.env_name))
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPG(state_dim, action_dim, cfg)
    return env, agent

def train(cfg, env, agent):
    ou_noise = OUNoise(env.action_space)
    rewards = []
    ma_rewards = []
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ou_noise.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1
            action = agent.choose_action(state)
            action = ou_noise.get_action(action, i_step)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state

        if (i_ep+1)%10 == 0:
            print('回合：{}/{}，奖励：{:.2f}'.format(i_ep+1, cfg.train_eps, ep_reward))

        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('完成训练！')
    return rewards, ma_rewards


def test(cfg, env, agent):
    print('开始测试！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = [] # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.test_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            env.render()
            i_step += 1
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
        print('回合：{}/{}, 奖励：{}'.format(i_ep+1, cfg.train_eps, ep_reward))
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{cfg.test_eps}，奖励：{ep_reward:.1f}")
    print('完成测试！')
    return rewards, ma_rewards

if __name__ == "__main__":
    cfg = DDPGConfig()
    plot_cfg = PlotConfig()
    env, agent = env_agent_config(cfg,seed=1)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(plot_cfg.result_path, plot_cfg.model_path)
    agent.save(path=plot_cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=plot_cfg.result_path)
    plot_rewards(rewards, ma_rewards, plot_cfg, tag="train")  # 画出结果
    # 测试
    env, agent = env_agent_config(cfg,seed=10)
    agent.load(path=plot_cfg.model_path)
    rewards, ma_rewards = test(cfg,env,agent)
    save_results(rewards, ma_rewards, tag = 'test', path = plot_cfg.result_path)
    plot_rewards(rewards, ma_rewards, plot_cfg, tag="test")  # 画出结果

