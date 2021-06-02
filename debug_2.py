import gym
import gym_oscillator
from gym import error, spaces, utils
from gym.utils import seeding
from gym_oscillator.envs.osc_env import oscillatorEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv,VecNormalize, VecEnv
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines import PPO2, ACKTR
import os
import numpy as np
import oscillator_cpp
import yaml
from collections import deque
import matplotlib.pyplot as plt
from stable_baselines.common import make_vec_env


if __name__ == "__main__":
    with open('configs/config.yml', 'r') as ymlfile:
        cfg = yaml.load(ymlfile,Loader=yaml.FullLoader)

    num_cpu = 1
#     TODO: maybe we need to define seeds for envs
    env = SubprocVecEnv([lambda: oscillatorEnv(cfg) for i in range(num_cpu)])
    env.reset()
    checkpoint_callback = CheckpointCallback(save_freq=10000000, save_path='./debug_400_PPO2/',                           name_prefix='rl_model_PPO2')
    model = PPO2(MlpPolicy, env, verbose=1,tensorboard_log="MLP/")
    model.learn(250000000, log_interval=1000, callback=checkpoint_callback )
    model.save('PPO2_180_5380_4_debug_400_episode_250mln')