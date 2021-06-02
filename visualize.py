import gym
import gym_oscillator
from gym import error, spaces, utils
from gym.utils import seeding
from gym_oscillator.envs.osc_env import oscillatorEnv

from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv,VecNormalize, VecEnv
from stable_baselines import PPO2, ACKTR

import os
import numpy as np
import oscillator_cpp
import yaml
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

with open('configs/config.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile,Loader=yaml.FullLoader)
    
model = ACKTR.load('ACKTR_180_5380_4_debug_400_episode_250mln.zip')

env = DummyVecEnv([lambda: oscillatorEnv(cfg)])
model.set_env(env)

N = 400 #steps to visiualize
num_steps = 1000

# best performance
rews_ = []
#Store observations
obs_ = []
acs_ = []
obs = env.reset()
cum_list = []
rew = 0
rews_before = []

for i in tqdm(range(num_steps)):
    action, _states = model.predict(obs)
    
    obs, rewards, dones, info = env.step(action)
    obs_.append(obs)
    rews_before.append(rewards[0]/100-1)
    acs_.append(action)
    rews_.append(rewards)
    rew += rewards
    
    if dones:
        obs = env.reset()
        cum_list.append(rew)
        rew = 0
        
# random performance
rews_r = []
#Store observations
obs_r = []
acs_r = []
obsr = env.reset()
cum_listr = []
rewr = 0
rews_beforer = []

for i in tqdm(range(num_steps)):
    action = model.env.action_space.sample()
    obs, rewards, dones, info = env.step(action)
    obs_r.append(obs)
    rews_beforer.append(rewards[0]/100-1)
    acs_r.append(action)
    rews_r.append(rewards)
    rewr += rewards
    
    if dones:
        obs = env.reset()
        cum_listr.append(rewr)
        rewr = 0

        
plt.title('Amplitude reduction')
plt.plot(rews_beforer[:N], 'o',label='random acts', color='r')
plt.plot( rews_before[:N],'o', label='neg. amplitude')
plt.legend()
plt.xlabel('theta')
plt.ylabel('amplitude')
plt.show()

def get_impulse(moment, a1, a2, a3, obs_,  rews_before):
    impulse = []
    a1, a2, a3 = a1[moment], a2[moment], a3[moment]
    theta = np.array(obs_).reshape(len(obs_))[moment]
    reward = rews_before[moment]
    Amp = 0.9
    full = 6320
    impulse = [0] + [Amp] * a1 + [0] * a2 + [-Amp/a3] * a1 * a3 + [0] * (full-a1-a2-a1*a3)
    plt.title('Optimal form of impulse for time {} and theta {}'.format(moment, round(float(theta), 2)))
    plt.plot(impulse, label='reward {}'.format(round(reward, 2)))
    plt.legend()
    plt.xlabel('amplitude')
    plt.ylabel('step')
    plt.show()

get_impulse(300, first, second, third, obs_,  rews_before)