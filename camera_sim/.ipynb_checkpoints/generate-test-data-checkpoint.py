import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import pickle
import math
import cv2
import random

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

random.seed(0)
npr.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Set up vars for saving
dir_path = "/home/anon/ahalya/LearningSensoryStructure/camera-intensity/data/rgb/rgb_test_data4/"


### Setup Franka Env ###
# Set up constants
tray_lim = np.array([[-0.2,0.2],[-0.3,-0.7]])
tray_ctr = np.mean(tray_lim, axis=1)

lim = [-1.,1.]
robot_lim = np.array([lim]*2)

# # Initialize Franka Robot
timeStep=1./60.
offset = [0.,0.,0.]
env = FrankaEnv(render=True, ts=timeStep,offset=offset)

### Move robot to starting pose
# xinit = [npr.uniform(lim[0],lim[1]), npr.uniform(lim[0],lim[1])]
xinit =  [-0.7815724015256085, 0.7432973423368283] 
env0 = ws_conversion(xinit, robot_lim, tray_lim)
cam_y = 0.3
r0 = [env0[0], cam_y, env0[1]]
pos = np.array(r0)
orn = env.bullet_client.getQuaternionFromEuler([math.pi/2.,0.,0.])
env.step(pos, orn)
input("Press enter")

obj_loc, _ = env.bullet_client.getBasePositionAndOrientation(env.objID)
obj2_loc, _ = None, None#env.bullet_client.getBasePositionAndOrientation(env.obj2ID)

test_traj = np.array([[0.,0.], [-0.1,0.], [-0.1,0.1],
             [0.,0.1], [0.1,0.1], [0.1,0.],
             [0.1,-0.1], [0., -0.1], [-0.1,-0.1]])*0.25#*0.75

path = []
env_path = []
data_buffer = []
img_num = 0
for iter_step in range(test_traj.shape[0]):
    state = test_traj[iter_step]
    path.append(state)

    ### Step in Franka environment ###
    # # Convert klerg state to environment state
    # ws_state = ws_conversion(state, robot_lim, tray_lim)
    ws_state = state
    # Generate target position in env
    pos = np.array([obj_loc[0] + ws_state[0], cam_y, obj_loc[2] + ws_state[1]])
    orn = env.bullet_client.getQuaternionFromEuler([math.pi/2.,0.,0.])
    env.step(pos, orn)
    env_path.append(pos)

    # Convert env state to klerg state
    robot_state = ws_conversion([pos[0], pos[2]], tray_lim, robot_lim)
    # Get camera image
    img = env.cam_img[:,:,:3]
    img = img.T
    
    # Subsample image
    data = np.array(img)
    data = data[:,::3,::3]
    data = data/255.0
    
    fname = dir_path + str(img_num)+'_test_img.png'
    plt.imsave(fname, data.T)

    # Push to buffer
    data_buffer.append((robot_state,data,iter_step))
    img_num +=1

# for iter_step in range(test_traj.shape[0]):
#     state = test_traj[iter_step]
#     path.append(state)

#     ### Step in Franka environment ###
#     # # Convert klerg state to environment state
#     # ws_state = ws_conversion(state, robot_lim, tray_lim)
#     ws_state = state
#     # Generate target position in env
#     pos = np.array([obj2_loc[0] + ws_state[0], cam_y, obj2_loc[2] + ws_state[1]])
#     orn = env.bullet_client.getQuaternionFromEuler([math.pi/2.,0.,0.])
#     env.step(pos, orn)
#     env_path.append(pos)

#     # Convert env state to klerg state
#     robot_state = ws_conversion([pos[0], pos[2]], tray_lim, robot_lim)
#     # Get camera image
#     img = env.cam_img[:,:,:3]
#     img = img.T
    
#     # Subsample image
#     data = np.array(img)
#     data = data[:,::3,::3]
#     data = data/255.0
    
#     fname = dir_path + str(img_num)+'_test_img.png'
#     plt.imsave(fname, data.T)

#     # Push to buffer
#     data_buffer.append((robot_state,data,iter_step))
#     img_num +=1

# print(img_num)

# Save Pickled Data
data_eval_dict = {
    "path": path,
    "buffer": data_buffer,
    "env_path": env_path,
    "obj_loc": obj_loc,
    "obj2_loc": obj2_loc,
    "tray_lim": tray_lim,
    "klerg_lim": robot_lim
    }
pickle.dump( data_eval_dict, open( dir_path+"data_eval_dict.pickle", "wb" ) )
