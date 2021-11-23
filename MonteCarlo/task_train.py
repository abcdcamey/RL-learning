# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2021/11/16 6:00 下午
# Description :
"""
import sys, os
import datetime
import torch
from ENVS.racetrack_env import RacetrackEnv
from MonteCarlo.agent import FirstVisitMC
from COMMON.utils import make_dir, save_results
from COMMON.plot import plot_rewards

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")

class MCconfig:
    def __init__(self):
        self.algo = "MC"
        self.env = "Racetrack"
        self.result_path = curr_path + "/outputs/" + self.env + \
                           '/' + curr_time + '/results/'  # path to save results
        self.model_path = curr_path + "/outputs/" + self.env + \
                          '/' + curr_time + '/models/'  # path to save models
        self.epsilon = 0.15
        self.gamma = 0.9
        self.train_eps = 200
        self.action_dim = 9
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def env_agent_config(cfg):
    """
    根据配置生成env和agent
    """
    env = RacetrackEnv()
    agent = FirstVisitMC(cfg)
    return env, agent

def train(cfg, env, agent):
    """
    训练
    """
    print("start to train")
    rewards = [] # 记录每个episode的reward
    ma_rewards = []
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0 #单episode获得的reward
        one_ep_transition = []
        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            ep_reward += reward
            one_ep_transition.append((state,action,reward))
            state = next_state
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        agent.update(one_ep_transition)
        if (i_ep+1)%10==0:
            print(f"Episode:{i_ep+1}/{cfg.train_eps};reward:{ep_reward}")
    print("Complete training")
    return rewards, ma_rewards



def eval(cfg, env, agent):
    print("Start to eval")
    print(f'Env:{cfg.env},Algo:{cfg.algo},Device:{cfg.device}')
    rewards = []
    ma_rewards = []
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        while True:
            action = agent.choose_action(state)
            env.render(0.2)
            next_state, reward, done = env.step(action)

            ep_reward += reward
            state = next_state
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep+1)%10 == 0:
            print(f"epicode:{i_ep+1}/{cfg.train_eps},reward:{ep_reward}")
    return rewards, ma_rewards

if __name__ == "__main__":

    curr_time = "20211117-104946"

    cfg = MCconfig()

    #train
    # env, agent = env_agent_config(cfg)
    # rewards, ma_rewards = train(cfg, env, agent)
    # make_dir(cfg.result_path, cfg.model_path)
    # agent.save(path=cfg.model_path)
    # save_results(rewards, ma_rewards, tag="train", path=cfg.result_path)
    # plot_rewards(rewards, ma_rewards, tag="train", env=cfg.env,
    #              algo=cfg.algo, path=cfg.result_path)

    curr_time = "20211117-104946"
    #eval
    env, agent = env_agent_config(cfg)
    agent.load(path=cfg.model_path)
    rewards, ma_rewards = eval(cfg, env, agent)

    save_results(rewards, ma_rewards, tag='eval', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, tag="eval", env=cfg.env, algo=cfg.algo, path=cfg.result_path)

