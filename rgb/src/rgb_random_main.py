import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import pickle
import math
# import cv2
import random
import os

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
save = False
dir_path = os.getcwd() + "/results/random/"
if not os.path.exists(dir_path): os.makedirs(dir_path)
print("Saving Directory Path: ", dir_path)

### Setup Franka Env ###
# Set up constants
tray_lim = np.array([[-0.2,0.2],[-0.3,-0.7]])
tray_ctr = np.mean(tray_lim, axis=1)

lim = [-1.,1.]
robot_lim = np.array([lim]*2)

# # Initialize Franka Robot
timeStep=1./60.
offset = [0.,0.,0.]
env = FrankaEnv(render=False, ts=timeStep,offset=offset)

### Move robot to starting pose
# xinit = [npr.uniform(lim[0],lim[1]), npr.uniform(lim[0],lim[1])]
xinit =  [-0.7815724015256085, 0.7432973423368283] 
env0 = ws_conversion(xinit, robot_lim, tray_lim)
cam_y = 0.3
r0 = [env0[0], cam_y, env0[1]]
pos = np.array(r0)
orn = env.bullet_client.getQuaternionFromEuler([math.pi/2.,0.,0.])
input("Press enter")

# # Initialize VAE & VAE buffer
model = VAE(input_dim=5120,  z_dim=6,s_dim=2,  img_dim=5,
            hidden_dim=16)#64)#16)
optimizer = optim.Adam(model.parameters(), lr=3e-3)
vae_buffer = ReplayBuffer(capacity=10000)
num_learning_opt = 10
batch_size = 10#10

mseloss = nn.MSELoss()

# Initialize KL-Erg Robot

dt = 0.1
n = int(dt/timeStep)

x0 = [xinit[0], xinit[1], 0,0]
path = []
env_path = []
actions = []
losses = []
data_buffer = []
num_steps = 1000

obj_loc = None
learning_ind=0
for iter_step in range(num_steps):
    ### Plan in KL-Erg ###
    state = npr.uniform(-1,1,2)

    ### Update lists ###    
    path.append(state)
    traj = np.array(path)

    ### Step in Franka environment ###
    # Convert klerg state to environment state
    ws_state = ws_conversion(state, robot_lim, tray_lim)

    # Generate target position in env
    pos = np.array([ws_state[0], cam_y, ws_state[1]])
    orn = env.bullet_client.getQuaternionFromEuler([math.pi/2.,0.,0.])
    env.step(pos, orn)

    ### Update data buffer ###
    # Get env state
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
    
    # Push to buffer
    vae_buffer.push(robot_state, data)
    data_buffer.append((robot_state,data))

    # Run VAE Optimization
    if iter_step > batch_size:
        if obj_loc is None:
            obj_loc, _ = env.bullet_client.getBasePositionAndOrientation(env.objID)
            print('obj location: {}'.format(obj_loc))

        print("iter_step: {}".format(iter_step))
        x_r = torch.FloatTensor(robot_state).unsqueeze(axis=0)
        y_r = torch.FloatTensor(data).unsqueeze(axis=0)

        for iter_opt in range(num_learning_opt):
            # Sample Buffer
            x, y = vae_buffer.sample(batch_size)
            x = torch.FloatTensor(x).squeeze()
            y = torch.FloatTensor(y).squeeze()
            # Run optimization
            y_pred, y_logvar, z_mu, z_logvar, z_samples = model(x, y)
            img_logvar = torch.ones(y_pred.shape)
            for i in range(y_pred.shape[0]):
                img_logvar[i,:,:,:] = y_logvar[i]
            p = Normal(z_mu, z_logvar.exp())
            q = Normal(y_pred, img_logvar.exp())

            loss = - torch.mean(q.log_prob(y)) \
                        - 0.1*  torch.mean(0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp()).sum(1)) #0.01
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            # print(loss.item())
            losses.append(loss.item())
            learning_ind += 1
        if save: 
            if learning_ind % 50 == 0:
                print('saving checkpoint: iter_step'.format(learning_ind))
                print(loss.item())
                cp_dict = {
                        'iter': learning_ind,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'xr': x_r, 
                        'y': y_r,
                }
                fpath = dir_path+ 'model_checkpoint_iter{}.pth'.format(learning_ind)
                torch.save(cp_dict, fpath)

    
path = np.array(path)
actions = np.array(actions)
env_path = np.array(env_path)
# Save Torch final model
torch.save(model, dir_path+'model_final.pth')
torch.save(optimizer, dir_path+'optim_final.pth')

# print("Final MSELoss: {}".format(mseloss(img_pred, y))

# Save Pickled Data
data_eval_dict = {
    "path": path,
    "actions": actions,
    "buffer": data_buffer,
    "env_path": env_path,
    "obj_loc": obj_loc,
    "losses": losses,
    "tray_lim": tray_lim,
    "klerg_lim": robot_lim
    }
pickle.dump( data_eval_dict, open( dir_path+"data_eval_dict.pickle", "wb" ) )

# Save pybullet environment
env.bullet_client.saveBullet(dir_path+"state.bullet")

for iter_opt in range(1000):
    # Sample Buffer
    x, y = vae_buffer.sample(batch_size)
    x = torch.FloatTensor(x).squeeze()
    y = torch.FloatTensor(y).squeeze()
    # Run optimization
    y_pred, y_logvar, z_mu, z_logvar, z_samples = model(x, y)
    img_logvar = torch.ones(y_pred.shape)
    for i in range(y_pred.shape[0]):
        img_logvar[i,:,:,:] = y_logvar[i]

    p = Normal(z_mu, z_logvar.exp())
    q = Normal(y_pred, img_logvar.exp())

    loss = - torch.mean(q.log_prob(y)) \
                - 0.1*  torch.mean(0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp()).sum(1)) #0.01
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    print(loss.item())
    losses.append(loss.item())
    learning_ind += 1
    if save: 
        if learning_ind % 50 == 0:
            print('saving checkpoint: iter_step'.format(learning_ind))
            print(loss.item())
            cp_dict = {
                    'iter': learning_ind,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'xr': x_r, 
                    'y': y_r,
            }
            fpath = dir_path+ 'model_checkpoint_iter{}.pth'.format(learning_ind)
            torch.save(cp_dict, fpath)

# Save Pickled Data
data_eval_dict = {
    "path": path,
    "actions": actions,
    "buffer": data_buffer,
    "env_path": env_path,
    "obj_loc": obj_loc,
    "losses": losses,
    "tray_lim": tray_lim,
    "klerg_lim": robot_lim
    }
pickle.dump( data_eval_dict, open( dir_path+"data_eval_dict_pl.pickle", "wb" ) )


# Save Torch final model
torch.save(model, dir_path+'model_final_postlearning.pth')
torch.save(optimizer, dir_path+'optim_final_postlearning.pth')


plt.plot(path[0,0], path[0,1], 'r*')
plt.plot(path[:,0], path[:,1], 'k.')
plt.axis("square")
plt.xlim([lim[0],lim[1]])
plt.ylim([lim[0],lim[1]])
plt.show()

plt.plot(env_path[0,0], env_path[0,2], 'r*')
plt.plot(env_path[:,0], env_path[:,2], 'k.')
plt.plot(obj_loc[0],obj_loc[2], 'b*')
plt.axis("square")
plt.xlim([tray_lim[0,0],tray_lim[0,1]])
plt.ylim([tray_lim[1,0],tray_lim[1,1]])
plt.show()
