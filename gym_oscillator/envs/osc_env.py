import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import oscillator_cpp
from gym import Env
from collections import deque
from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box, Bbox, DBox
import matplotlib.pyplot as plt
# from gym.spaces import DBox
# from gym_oscillator.envs.space import Bbox, BBbox


class oscillatorEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, config):
#         TODO update finally description
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
        self.a_0 = np.array([config['width_p'], config['gap'], config['Kfactor']], dtype=int)
        self.R0 = config['R0']
        self.theta0 = config['theta_0']
        self.width_p = config['width_p']
        self.gap = config['gap']
        self.Kfactor = config['Kfactor']
        self.ep_length = config['ep_length']
        self.R = config['R0']
        self.theta = config['theta_0']
        act_space = config['action_space']
        self.steps = config['step']
             
        # Call init function and save params
        self.state = oscillator_cpp.init(self.ep_length, self.R0, self.theta0, self.width_p, self.gap, self.Kfactor) 
        # Limits for actions
#         self.high_a = np.array([1256, 6278, 4]) 
#         self.high_a = np.array([900, 3000, 2]) #second
#         self.high_a = np.array([180, 5380, 4])
        self.interval = config['interval_theta'] #[0, pi/2) , [pi/2, pi], [pi, 3pi/2], [3pi/2, 2pi]
        theta_intervals = [[0, np.pi/2], [np.pi/2, np.pi], [np.pi, 3*np.pi/2], [3*np.pi/2, 2*np.pi], [0, 2*np.pi]]
        
        self.high_a = np.array(config['high_actions']) 
        self.low_a = np.array(config['low_actions'])
#         print(high_a,low_a)
        self.action_shape = self.high_a.shape[0]
        self.low_state = theta_intervals[self.interval][0]
        self.high_state = theta_intervals[self.interval][1]
        
        if act_space == 'Bbox':
            denorm_limits = [self.low_a, self.high_a]
            self.action_space = Bbox(low=-1, high=1, act_limits = denorm_limits, shape=(self.action_shape,),)
            print(self.action_space)
        elif act_space == 'DBox':
            denorm_limits = [self.low_a, self.high_a]
            self.action_space = DBox(low=-1, high=1, act_limits = denorm_limits, shape=(self.action_shape,),)
        else:
            self.action_space = Box(low=-1, high=1, shape=(self.action_shape,),)
           
            
        self.observation_space = Box(low=self.low_state, high=self.high_state, shape=(1,), dtype=np.float32) #[theta_t] 
        self.done = False
        self.current_step = 0   
        # Memory for samples
#         self.memory = deque(maxlen=config['memory_size'])
        #Reset environment
        print('envs', self.action_space)
        self.reset()
        
    def denormalize_actions(self, action):
#         print('denorm in func', action, self.low_a, self.high_a)
        de_act = self.low_a + (self.high_a - self.low_a) * (action + 1) / 2 
#         print('result', de_act)
        return de_act
  
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
#         self.width_p, self.gap, self.Kfactor = int(de_actions[0]), int(de_actions[1]), int(de_actions[2])
        self.gap, self.Kfactor = int(de_actions[0]), int(de_actions[1])
#         print('denorm', self.width_p, self.gap, self.Kfactor)
        value = self.width_p + self.gap + self.width_p * self.Kfactor
        assert value <= 6280
        self.theta = self.low_state + (self.high_state - self.low_state) / self.ep_length * self.current_step 
#         print(self.theta, self.current_step)
# self.R=R0=1 the same
        self.state = oscillator_cpp.collect_state(self.R, self.theta, self.state)
        self.state = oscillator_cpp.Make_step(self.state, self.width_p, self.gap, self.Kfactor)
        # calculate reward
        reward = self.Reward(oscillator_cpp.Calc_x(self.state))  
#         print(reward)
        self.current_step += self.steps
        self.done = self.current_step >= self.ep_length
        return np.array([self.theta]), reward, self.done, {} 

    def reset(self):
        self.current_step = 0 
        self.theta = self.low_state
        self.R = 1
        self.state = oscillator_cpp.init(self.ep_length, self.R0, self.theta0, int(self.a_0[0]), int(self.a_0[1]), int(self.a_0[2]))
        self.done = False
        return np.array([self.theta])
    
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
        reward = (-R + 1) * 100
        return reward