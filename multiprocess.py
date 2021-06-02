import gym
import pandas as pd
import gym_oscillator
import oscillator_cpp
from stable_baselines.common import set_global_seeds
from IPython.display import clear_output
from stable_baselines.common.policies import MlpPolicy, LstmPolicy, CnnPolicy,MlpLstmPolicy,ActorCriticPolicy
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv,VecNormalize, VecEnv

from stable_baselines import PPO2, DDPG, DQN, PPO1, A2C, ACKTR, SAC
from collections import OrderedDict
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info
from multiprocessing import Pool
import multiprocessing


import logging
from logging.handlers import QueueHandler, QueueListener
import time

def make_env(env_id, rank, seed=0,s1=False,s2=False,s3=False,s4=False,s5=False,skip=2):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        print(env.reset().shape)
        env.reward_init(s1,s2,s3,s4,s5,skip = skip)
        return env
    set_global_seeds(seed)
    return _init



def worker_init(q):
    # all records from worker processes go to qh and then into q
    qh = QueueHandler(q)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(qh)


def logger_init():
    q = multiprocessing.Queue()
    # this is the handler for all log records
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(asctime)s - %(process)s - %(message)s"))

    # ql gets records from the queue and sends them to the handler
    ql = QueueListener(q, handler)
    ql.start()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # add the handler to the logger so records from this process are handled
    logger.addHandler(handler)

    return ql, q


def get_stds_and_evaluate(coupling_power):
    skip_rate=0
    neurons_number=1000
    before_range=5000000
    after_range=5000000
    file_name='Default'+'_'+str(coupling_power)+'_' + str(time.time())
    num_cpu = 1
    env = DummyVecEnv([make_env('oscillator-v0', i,s2=True) 
                            for i in range(num_cpu)])
    model = PPO2(MlpPolicy, env, verbose=0)
    model = model.load('trained_models/Ps6_final_3')
    env = gym.make('oscillator-v0',)
    env.__init__(epsilon=coupling_power,nosc=neurons_number)
    s1,s2,s3,s4,s5 = False,True,False,False,False
    env.reward_init(s1,s2,s3,s4,s5,skip=skip_rate)
        # The algorithms require a vectorized environment to run
    rews_ = []
    obs_ = []
    obs = env.reset()
    acs_ = []
    states_x = []
    states_y = []

    for i in range(before_range):
        if (i % 50000) == 0:
            logging.info('Before function called itteration {} in worker thread with coupling {}'.format(i,coupling_power),)
        obs, rewards, dones, info = env.step([0])
        states_x.append(env.x_val)
        states_y.append(env.y_val)
        obs_.append(obs[0])
        acs_.append(0)
        rews_.append(rewards)


    for i in range(after_range):
        if (i % 50000) == 0:
            logging.info('After function called itteration {} in worker thread with coupling {}'.format(i,coupling_power),)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        states_x.append(env.x_val)
        states_y.append(env.y_val)
        obs_.append(obs[0])
        acs_.append(env.val)
        rews_.append(rewards)
        
    for i in range(1000):
        obs, rewards, dones, info = env.step([0])
        states_x.append(env.x_val)
        states_y.append(env.y_val)
        obs_.append(obs[0])
        acs_.append(0)
        rews_.append(rewards)

    
    #pd.DataFrame([states_x,acs_]).T.to_csv(file_name+'.csv')    
    before_std = np.std(states_x[:before_range])
    before_mean = np.mean(states_x[:before_range])
    after_std = np.std(states_x[before_range:after_range+before_range])
    after_mean = np.mean(states_x[before_range:after_range+before_range])
    return (states_x,acs_)
    

    
def main():
    q_listener, q = logger_init()
    coupling_numbers = [0.02]
    coupling_stds = {}
    results = get_stds_and_evaluate(0.02)
    
    out_ = pd.DataFrame([results[0],results[1]]).T.to_csv(str(time.time())+'_'+'test_mpl_chaos.csv')

if __name__ == '__main__':
    main()
