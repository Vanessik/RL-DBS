import gym
import gym_oscillator
from gym import error, spaces, utils
from gym import Space
from gym.utils import seeding
from gym_oscillator.envs.osc_env import oscillatorEnv

from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv,VecNormalize, VecEnv
from stable_baselines import PPO2, ACKTR
import numpy as np
from gym.spaces import Box
import os
import numpy as np
import oscillator_cpp
import yaml
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools.tools import show_statistics


# File for statistics computation

if __name__ == "__main__":
    with open('configs/config.yml', 'r') as ymlfile:
        cfg = yaml.load(ymlfile,Loader=yaml.FullLoader)

#     num_cpu = 12
#     TODO: maybe we need to define seeds for envs
    env = DummyVecEnv([lambda: oscillatorEnv(cfg)])
    env.reset()
    model = ACKTR(MlpPolicy, env, verbose=1,tensorboard_log="MLP/")
    model.learn(10, log_interval=1000)

    normal_vec = []
    for j in tqdm(range(10000)):
        rews_r = []
        #Store observations
        obs_r = []
        acs_r = []
        obsr = env.reset()
        cum_listr = []
        rewr = 0
        rews_beforer = []

        for i in range(400):
            action = np.array(model.env.action_space.sample())
            obs, rewards, dones, info = env.step(action)
            obs_r.append(obs)
            rews_beforer.append(rewards[0]/100-1)
            acs_r.append(action)
            rews_r.append(rewards)
            rewr += rewards
            if (rewards[0]/100-1)>-0.97:
                normal_vec.append([obs, rewards[0]/100-1, action])

            if dones:
                obs = env.reset()
                cum_listr.append(rewr)
                rewr = 0
    for i in range(4):
        print('{} interval'.format(i), file=open("output.txt", "a"))
        print(show_statistics(i, normal_vec, model, True), file=open("output.txt", "a"))
          

