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

# Set up vars for saving
dir_path = "/home/anon/ahalya/LearningSensoryStructure/franka-sim/data/misc/"


### Setup Franka Env ###
# Set up constants
tray_wslims = np.array([[-0.2,0.2],[-0.4,-0.8]])
tray_vertices = np.array([[-0.2,-0.4],[0.2,-0.4], [0.2,-0.8],[-0.2,-0.8]])
tray_ctr = np.mean(tray_wslims, axis=1)

lim = [0.,1.]
# # Initialize Franka Robot
timeStep=1./60.
offset = [0.,0.,0.]
env = FrankaEnv(render=True, ts=timeStep,offset=offset)


# Initialize KL-Erg Robot

dt = 0.1
n = int(dt/timeStep)
x0 = [npr.uniform(lim[0],lim[1]), npr.uniform(lim[0],lim[1]), 0,0]

indices = []
path = []
env_path = []
actions = []
losses = []
data_buffer = []
num_steps = 30

obj_loc = None
for iter_step in range(num_steps):
    obj_loc, _ = env.bullet_client.getBasePositionAndOrientation(env.objID)

    ### Step in Franka environment ###
    # Convert search state to robot state
    pos = np.array([obj_loc[0], 0.6, obj_loc[2]])
    orn = env.bullet_client.getQuaternionFromEuler([math.pi/2.,0.,0.])
    # Simulate in Franka env
    for j in range(n):
        env.step(pos, orn)



    print('obj location: {}, robot location: {}'.format(obj_loc, pos))
    if iter_step > 20:
        print("start saving")

            
        print('obj location: {}'.format(obj_loc))
        # Update data buffer
        indices.append(iter_step)
        pos, orn = env.get_ee_state()
        env_path.append(pos)
        state = [pos[0], pos[2]]
        path.append(state)
        img = env.cam_img[:,:,:3]

        fname = dir_path+str(iter_step)+'_img.png'
        cv2.imwrite(fname, img)

        img = img.T
        data = np.array(img)
        data_buffer.append((state,data, iter_step))



# Save Pickled Data
img_dict = {
    "path": path,
    "buffer": data_buffer,
    "env_path": env_path,
    "obj_loc": obj_loc,
    "indices": indices
    }
pickle.dump(img_dict, open( dir_path+"data_dict.pickle", "wb" ) )

# Save pybullet environment
# env.bullet_client.saveBullet(dir_path+"state.bullet")

plt.plot(path[0,0], path[0,1], 'r*')
plt.plot(path[:,0], path[:,1], 'k.')
plt.axis("square")
plt.xlim([lim[0],lim[1]])
plt.ylim([lim[0],lim[1]])
plt.show()
