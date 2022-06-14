import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import pickle
import os

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import Normal

from klerg_main import Robot
from electro_utils import *
from vae import VAE
from vae_buffer import ReplayBuffer

# Set up vars for saving
save = False
dir_path = os.getcwd() + "/results/ergodic/"
if not os.path.exists(dir_path): os.makedirs(dir_path)
print(dir_path)
# Set up Environment
lim = [-1.,1.]
npr.seed(666)
obj_list = npr.uniform(lim[0], lim[1], (1,2))
# obj_loc = np.array([-0.49106524,  0.12775323]) # location for ent1, unifopt1, rand1
# obj_loc = np.array([0.56700174,  0.64450871]) # location for ent2, unifopt2, rand2
# obj_list = [obj_loc]
# Set up measurement models

data_model = MeasurementModel()
y_actual = MeasurementModel_nonoise()

# Set up VAE and VAE buffer
model = VAE(input_dim=data_model.n, output_dim=data_model.n, z_dim=2, s_dim=2, hidden_dim=64)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
vae_buffer = ReplayBuffer(capacity=10000)
num_learning_opt = 10
batch_size = 64

# Set up Klerg Robot
x0 = [npr.uniform(lim[0], lim[1]), npr.uniform(lim[0],lim[1]), 0,0]
robot = Robot(x0=x0, wslim=lim, ts=0.1, explr_idx=[0,1], target_dist=model)
path = []
num_steps = 1000
num_env = len(obj_list)
actions = []
losses = []
data_buffer = []
opt_num = 0

for env_num in range(num_env):
    obj_loc = obj_list[env_num]
    print(env_num, obj_loc)
    model.init = False
    for i in range(num_steps):
        # Run Klerg Optimization
        state, action = robot.step(num_target_samples= 100, num_traj_samples=num_steps, R=0.01, alpha=1e-3)
        path.append(state)
        actions.append(action)
        traj = np.array(path)
        # Update data buffer

        data = data_model(np.array([obj_loc-state[:2]]))
        data_buffer.append((state, data))
        vae_buffer.push(state[:2], data)
        x_r = torch.FloatTensor(state[:2]).unsqueeze(axis=0)
        y_r = torch.FloatTensor(data)
        # Run VAE Optimization
        if len(vae_buffer) > batch_size:
            print("i: {}".format(i))
            for iter_opt in range(num_learning_opt):
                # Sample Buffer
                x, y = vae_buffer.sample(batch_size)
                x = torch.FloatTensor(x).squeeze()
                y = torch.FloatTensor(y).squeeze()
                # Run optimization
                y_pred, y_logvar, z_mu, z_logvar, z_samples = model(x, y)
                p = Normal(z_mu, z_logvar.exp())
                q = Normal(y_pred, y_logvar.exp().repeat(1,y_pred.shape[1]))
                loss = - torch.mean(q.log_prob(y)) \
                            - 0.1*  torch.mean(0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp()).sum(1)) #0.01
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                print(loss.item())
                losses.append(loss.item())
            if i > 32: 
                model.update_dist(x_r, y_r) # comment out for unif klerg and random sampling
            if save: 
                if (i % 5) == 0:
                    cp_dict = {
                            'iter': i,
                            'env_num': env_num,
                            'opt_num': opt_num,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'xr': x_r, 
                            'y': y_r,
                    }
                    fpath = dir_path+ 'model_checkpoint_iter{}.pth'.format(opt_num)
                    torch.save(cp_dict, fpath)
            opt_num += 1
            # plt.pause(0.2)

path = np.array(path)
actions = np.array(actions)

torch.save(model, dir_path+'model_final.pth')
torch.save(optimizer, dir_path+'optim_final.pth')

data_eval_dict = {
    'obj_list': obj_list,
    'traj': traj,
    'path': path,
    'actions': actions,
    'loss': losses,
    'data_buffer': data_buffer

}
pickle_out = open(dir_path+"data_dict.pickle", "wb")
pickle.dump(data_eval_dict, pickle_out)
pickle_out.close()

for iter_opt in range(1000):
    # Sample Buffer
    x, y = vae_buffer.sample(batch_size)
    x = torch.FloatTensor(x).squeeze()
    y = torch.FloatTensor(y).squeeze()
    # Run optimization
    y_pred, y_logvar, z_mu, z_logvar, z_samples = model(x, y)
    p = Normal(z_mu, z_logvar.exp())
    q = Normal(y_pred, y_logvar.exp().repeat(1,y_pred.shape[1]))
    loss = - torch.mean(q.log_prob(y)) \
                - 0.1*  torch.mean(0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp()).sum(1)) #0.01
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    print(loss.item())
    losses.append(loss.item())

torch.save(model, dir_path+'model_final_postlearning.pth')
torch.save(optimizer, dir_path+'optim_final_postlearning.pth')

print(obj_loc)

xc, yc = np.meshgrid(np.linspace(lim[0],lim[1],21), 
                       np.linspace(lim[0],lim[1],21))
plt.close()
samples = np.c_[xc.ravel(), yc.ravel()]
data = [None]*len(samples)
for i in range(len(samples)):
    xr = np.expand_dims(samples[i], 0)
    y = data_model(obj_loc-xr)

    xr = torch.FloatTensor(xr)
    y = torch.FloatTensor(y)
    y_pred, y_logvar, z_mu, z_logvar, z_samples = model(xr, y)

    y_out = model.decode(torch.cat([z_samples, xr], dim=1))
    y_pred, _ = y_out[:, :model.output_dim], y_out[:, model.output_dim:]
    data[i] = y_pred[:,221].detach().numpy()
data = np.reshape(data, (21,21))
plt.contourf(xc, yc, data)
plt.axis('square')
plt.plot(obj_loc[0],obj_loc[1], 'm*')
plt.plot(traj[0,0], traj[0,1], 'r*')
plt.plot(traj[:,0], traj[:,1], 'k.')
plt.show()
