"""
This script is the main training script.

Currently it takes in several command line arguments:

-tb:  A path where the tensorboard information will be saved (reward, etc)
-n:  The number of timesteps to train for (default is set in config.yml)
-s:  A string that will form the filename of the saved file
-ncpu: The number if cpus to train on
-f: Frequency saving callbacks
-save: Should we save model

It then builds the environment, policy network, trains the agent, and saves the trained model.
"""
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
import argparse
import numpy as np
import oscillator_cpp
import yaml
from collections import deque
import matplotlib.pyplot as plt
from stable_baselines.common import make_vec_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-tb', dest='tensorboard_log_dir',
        help='Tensorboard log dir', default='MLP/', type=str)
    parser.add_argument('-n',dest='num_learning_steps',
        help='Overwrite the number of learning steps', default=10000000,type=int)
    parser.add_argument('-s',dest='model_name',
        help='Save the trained model here',default='./model_new/ACKTR_new',type=str)
    parser.add_argument('-ncpu',dest='num_cpu',
        help='Number of cpu for train',default=12,type=int)
    parser.add_argument('-f',dest='save_freq',
        help='Frequency saving callbacks',default=100000,type=int)
    parser.add_argument('-save',dest='save',
        help='Should we save our trained model',default=True,type=bool)
    
    args = parser.parse_args()
    with open('configs/config.yml', 'r') as ymlfile:
        cfg = yaml.load(ymlfile,Loader=yaml.FullLoader)

    env = SubprocVecEnv([lambda: oscillatorEnv(cfg) for i in range(args.num_cpu)])
    env.reset()
    checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path='./logs/',                           name_prefix='rl_model_ACKTR')
    model = ACKTR(MlpPolicy, env, verbose=1,tensorboard_log=args.tensorboard_log_dir)
    model.learn(args.num_learning_steps, log_interval=args.save_freq, callback=checkpoint_callback )
    if args.save:
        model.save(args.model_name)    
    