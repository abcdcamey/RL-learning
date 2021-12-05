# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2021/12/1 7:47 下午
# Description : 
"""
import sys, os
import datetime
import torch
import gym
from DQN.agent import DQN
from COMMON.utils import save_results, make_dir
from COMMON.plot import plot_rewards,plot_rewards_cn

curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
sys.path.append(parent_path) # 添加父路径到系统路径sys.path


curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class DQNConfig:
    def __init__(self):
        self.algo = "DQN"
        self.env = "CartPole-v0"
        self.result_path = curr_path + "/outputs/" + self.env + \
                           '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env + \
                          '/' + curr_time + '/models/'  # 保存模型的路径
        self.train_eps = 200
        self.eval_eps = 50
        self.gamma = 0.95
        self.epsilon_start = 0.9 #e-greedy策略中初始的e
        self.epsilon_end = 0.01
        self.epsilon_decay = 500

        self.lr = 0.0001
        self.memory_capacity = 100000
        self.batch_size = 64
        self.target_update = 4
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.hidden_dim = 256


def env_agent_config(cfg, seed=1):
    env = gym.make(cfg.env)
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(cfg, state_dim, action_dim)
    return env, agent


def train(cfg, env, agent):
    rewards = []
    ma_rewards = []
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            agent.update()
            if done:
                break
        if (i_ep+1) % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        if (i_ep+1)%10 == 0:
            print('回合：{}/{}, 奖励：{}'.format(i_ep+1, cfg.train_eps, ep_reward))
        rewards.append(ep_reward)

        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)

    print("训练完成")
    return rewards, ma_rewards

def eval(cfg, env, agent):
    rewards = []
    ma_rewards = []
    for i_ep in range(cfg.eval_eps):
        ep_reward = 0
        state = env.reset()
        while True:
            env.render()
            action = agent.predict(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9 + ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f"回合：{i_ep + 1}/{cfg.eval_eps}, 奖励：{ep_reward:.1f}")
    print('完成测试！')
    return rewards, ma_rewards


if __name__ == "__main__":
    cfg = DQNConfig()
    env, agent = env_agent_config(cfg, seed=1)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards, ma_rewards, tag="train", path = cfg.result_path)
    plot_rewards_cn(rewards, ma_rewards, tag="train",
                    algo=cfg.algo, path=cfg.result_path)

    # 测试
    env, agent = env_agent_config(cfg, seed=10)
    agent.load(path=cfg.model_path)
    rewards, ma_rewards = eval(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='eval', path=cfg.result_path)
    plot_rewards_cn(rewards, ma_rewards, tag="eval", env=cfg.env, algo=cfg.algo, path=cfg.result_path)