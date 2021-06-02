import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import oscillator_cpp
from gym import Env
from collections import deque
from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box
import matplotlib.pyplot as plt
# import sys
# path_now = sys.path[-1]
# path_to_file = '/home/v_skliarova/RL-DBS/one_step/debug/gym_oscillator/envs'
# sys.path.append(path_to_file)
# from space import Bbox
# sys.path.append(path_now)
#TODO 
# visualization maybe in different file
# Wrapper for callbacks during training

class oscillatorEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, config):
        """
        Goal: minimize R after one step of ARC function;
        Inputs:
        ep_length: Number of points for ARC: integer = episode length;
        -----------
        Oscillator params:
        R0:  cycle amplitude: double;
        width_p: Width of the 1st pulse in steps: integer;
        gap: Gap in steps: integer;
        Kfactor: Ratio of the 2nd width over 1st width: double;
        ------------    
        Observation:
            Type: Box(1)
            Num     Observation       Min                     Max
            0       theta              0                      2*np.pi
        Actions:
            Type: Box(3)
            Num     Observation       Min                     Max
            0       width_p           1                       inf
            1       gap               0                       inf
            2       Kfactor           1                       20
        --------------
        Reward:
        Reward defined to minimize the amplitude R.
        --------------
        Starting State:
        Each s_0 = [R_0, theta_0] = [1, 0]
        s_t = [R_0, theta_t] = [1,  2*pi/ep_length*current_step]
        --------------
        Episode Termination:
        After ep_length steps(number of limit cycles).
        """
        super(oscillatorEnv, self).__init__()
        #initial conditions
        self.a_0 = np.array([config['width_p'], config['gap'],  config['Kfactor']], dtype=int)
        self.R0 = config['R0']
        self.theta0 = config['theta_0']
        self.width_p = config['width_p']
        self.gap = config['gap']
        self.Kfactor = config['Kfactor']
        self.ep_length = config['ep_length']
     
        # Call init function and save params
        self.R = config['R0']
        self.theta = config['theta_0']
        self.state = oscillator_cpp.init(self.ep_length, self.R0, self.theta0, self.width_p, self.gap, self.Kfactor) 
        # Limits for actions
        self.high_a = np.array([180, 5380, 4]) # np.array([1256, 6278, 4])
        self.low_a = np.array([1, 0, 1])
        
        self.action_space = Box(low=-1, high=1, shape=(3,),)
        self.observation_space = Box(low=np.array([0.1, 0]), high=np.array([1, 2*np.pi]), shape=(2,), dtype=np.float32) #[theta_t] 
        self.done = False
        self.current_step = 0   
        # Memory for samples
        self.memory = deque(maxlen=config['memory_size'])
        #Reset environment
        self.reset()
        
    def denormalize_actions(self, action):
        de_act = self.low_a + (self.high_a-self.low_a)*(action+1)/2 
        return de_act
    
#     Add visualization
    def visualize_actions(self, actions_array):
        denorm_act = [self.denormalize_actions(i) for i in actions_array]
        width_p_ = [i[0] for i in denorm_act]
        gap_ = [i[1] for i in denorm_act]
        Kfactor_ = [i[2] for i in denorm_act]
        fig, axs = plt.subplots(3, 1)
        fig.suptitle('Actions visualization', fontsize='large')
        axs[0].plot(width_p_)
        axs[0].set_ylabel('width_p')
        axs[1].plot(gap_)
        axs[1].set_ylabel('gap')
        axs[2].plot(Kfactor_)
        axs[2].set_xlabel('step')
        axs[2].set_ylabel('Kfactor')
        fig.tight_layout()
        plt.show()
        
    def visualize_state(self, states):
        plt.title('Theta state visualization')
        plt.plot(states)
        plt.show()
        
    def visualize_rewards(self, rewards):
        plt.title('Rewards visualizations')
        plt.plot(rewards)
        plt.show()
  
    def record(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def step(self, action):
        """
        Function that called at each step.
        -----------
        Inputs:
        action: signal to make step: [[float]]
        -----------
        Returns:
        self.state: Vector of our state: np.array
        reward: Our reward function: float, 
        self.done: Does it end?: Bool, 
        additional_information: Nothing to show :( :{} 
        """
        #x = (y-y1)/(y2-y1)*(x2-x1)+x1
        de_actions = self.denormalize_actions(action) 
        self.width_p, self.gap, self.Kfactor = int(de_actions[0]), int(de_actions[1]), int(de_actions[2])
        # create state
#         print('dd', int(de_actions[0]), round(de_actions))
        value=self.width_p+self.gap+self.width_p*self.Kfactor
#         print('ddd', value,2*np.pi/0.001)  #6283
#         print('cur', self.current_step)
        self.theta = 2*np.pi/self.ep_length*self.current_step 
# self.R=R0=1 the same
        self.state = oscillator_cpp.collect_state(self.R, self.theta, self.state)
        self.state = oscillator_cpp.Make_step(self.state, self.width_p, self.gap, self.Kfactor)
        # calculate reward
        reward = self.Reward(oscillator_cpp.Calc_x(self.state))  
        self.current_step += 1
        self.done = self.current_step >= self.ep_length
#         print('was', self.R)
        return_state = np.array([self.R, self.theta])
        self.R=oscillator_cpp.Calc_x(self.state)
#         print('bec', self.R)
        return return_state, reward, self.done, {} 

    def reset(self):
        self.current_step = 0 
        self.theta = 0
        self.R = 1
        self.state = oscillator_cpp.init(self.ep_length, self.R0, self.theta0, int(self.a_0[0]), int(self.a_0[1]), int(self.a_0[2]))
        self.done = False
        return np.array([self.R, self.theta])
    
    def render(self, mode='human', close=False):
        pass

    def Reward(self, R):
        """
        Inputs:
        R: amplitude, that we want to minimize: float
        ------------
        Returns: 
        Reward: float
        """
        reward = (-R+1)*100
        return reward