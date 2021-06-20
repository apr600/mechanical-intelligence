import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import pickle
import math
import cv2

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import Normal

from klerg.klerg import Robot
from franka.franka_env import FrankaEnv
from vae.vae import VAE
from vae.vae_buffer import ReplayBuffer
from franka.franka_utils import *
import random
# Set up vars for saving
# dir_path = "/home/anon/ahalya/LearningSensoryStructure/franka-sim/data/entklerg1/"
random.seed(0)
npr.seed(0)
np.random.seed(0)
torch.manual_seed(0)

### Setup Franka Env ###

lim = [-1.,1.]

# Initialize KL-Erg Robot

dt = 0.1
# n = int(dt/timeStep)
x0 = [npr.uniform(lim[0],lim[1]), npr.uniform(lim[0],lim[1]), 0,0]
#  [0.23675041559133891, 0.9710395976790476, 0, 0]
# x0 [0.7815724015256085, 0.7432973423368283, 0, 0]** 

robot = Robot(x0=x0, wslim=lim, ts=dt, explr_idx=[0,1], buffer_capacity=10000)#,  target_dist=model)

path = []
env_path = []
actions = []
losses = []
data_buffer = []
num_steps = 1000

for iter_step in range(num_steps):
    ### Step in KL-Erg ###
    # state, action = robot.step() # default for input target dist
    # state, action = robot.step(num_target_samples= 100, num_traj_samples=150, R=3., alpha=1e-3) # Optimized for uniform
    
    state, action = robot.step(num_target_samples= 100, num_traj_samples=num_steps, R=0.5, alpha=1e-3)
    path.append(state)
    actions.append(action)
    traj = np.array(path)
    
    
path = np.array(path)
actions = np.array(actions)

plt.plot(path[0,0], path[0,1], 'r*')
plt.plot(path[:,0], path[:,1], 'k.')
plt.axis("square")
plt.xlim([lim[0],lim[1]])
plt.ylim([lim[0],lim[1]])
plt.show()

print("x0", x0)
