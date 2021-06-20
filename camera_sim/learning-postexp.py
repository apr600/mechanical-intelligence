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



# env_lim = np.array([[-0.2,0.2],[-0.3,-0.7]])
# klerg_lim = np.array([[-1., 1.],[-1.,1.]])

# Load collected data
# Load Pickled Data
dir_path1 = "data/entklerg12/"
# Load Pickled Data
file_path1 = dir_path1 + "data_eval_dict.pickle"
with open(file_path1, 'rb') as f: 
    data_dict1 = pickle.load(f, encoding="bytes")
    print(data_dict1.keys())
obj1 = data_dict1['obj_loc']
obj2 = data_dict1['obj2_loc']
traj1 = data_dict1['path']
loss1 = data_dict1['losses']
buffer1 = data_dict1['buffer']
action1 = data_dict1['actions']
env_traj1 = data_dict1['env_path']
env_lim = data_dict1['tray_lim']
klerg_lim = data_dict1['klerg_lim']

# # Load Pytorch Models
# model = torch.load(dir_path1 + "model_final.pth")
# model.eval()

# # Initialize VAE & VAE buffer
model = VAE(input_dim=5120,  z_dim=27,s_dim=2,  img_dim=5,
            hidden_dim=64)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_learning_opt = 5000
batch_size = len(buffer1)


print(env_lim, klerg_lim)
env_traj1 = np.array(env_traj1)
obj1 = np.array(obj1)

obj_ss = ws_conversion(np.array([obj1[0], obj1[2]]), env_lim, klerg_lim)
print(obj1, obj_ss)

plt.plot(env_traj1[:,0], env_traj1[:,2],'b.')
plt.plot(obj1[0], obj1[2],'k.')
plt.axis('square')
plt.xlim(env_lim[0])
plt.ylim(env_lim[1])
plt.show()


xlist = np.linspace(klerg_lim[0,0],klerg_lim[0,1],21)
ylist = np.linspace(klerg_lim[1,0],klerg_lim[1,1],21)

xx, yy = np.meshgrid(xlist, ylist)
pts = np.array([xx.ravel(), yy.ravel()]).T

vae_buffer = np.array(buffer1)
for iter_opt in range(num_learning_opt):
    # Sample Buffer
    batch = np.array(vae_buffer)[np.random.choice(len(vae_buffer),
                batch_size, replace=False)]
    x, y = map(np.stack, zip(*batch))
    # for ind in range(len(x)):
    #     x[ind] = np.array(ws_conversion(x[ind], env_lim, klerg_lim))
    x = torch.FloatTensor(x).squeeze()
    y = torch.FloatTensor(y).squeeze()

    # Run optimization
    y_pred, img_logvar_scalar, z_mu, z_logvar, z_samples = model(x,y)
    ysize = y_pred.size()
    img_logvar = img_logvar_scalar.unsqueeze(2).unsqueeze(3).expand(ysize)

    p = Normal(z_mu, z_logvar.exp())
    q = Normal(y_pred, img_logvar.exp())

    loss = - torch.mean(q.log_prob(y)) \
                - .1*  torch.mean(0.5 * (1 + z_logvar - z_mu.pow(2)
                - z_logvar.exp()).sum(1)) #0.01
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    if iter_opt %100 ==0:
        print(iter_opt, loss.item())

        # with torch.no_grad():
        #     sample = vae_buffer[np.random.choice(vae_buffer.shape[0], 1, replace=False)]
        #     x, y = map(np.stack, zip(*sample))
        #     for ind in range(len(x)):
        #         x[ind] = np.array(ws_conversion(x[ind], env_lim,
        #                                         klerg_lim))

        #         x = torch.FloatTensor(x)#.squeeze()
        #         y = torch.FloatTensor(y)#.squeeze()

        #         x_obj = torch.FloatTensor(np.expand_dims([obj1[0],
        #                                 obj1[2]],axis=0))

        #         _,_, z_mu, z_logvar, z_samples = model(x, y)
        #         z_samples = model.reparameterize(z_mu, z_logvar)

        #     var_list = []
        #     for i in range(len(pts)):
        #         x = torch.FloatTensor(np.expand_dims(pts[i],
        #                                          axis=0))
        #         img_logvar = model.imgvar_decoder(torch.cat([z_samples, x],
        #                                              dim=1))
        #         # img_logvar = torch.clamp(img_logvar, -5, 2)
        #         var_list.append(img_logvar[0].exp().detach().numpy())
        # var_list = np.array(var_list)
        # plt.contourf(xx,yy,var_list.reshape(xx.shape))
        # plt.plot(obj_ss[0], obj_ss[1],'k*')
        # plt.pause(1)


# # Reconstruct img prediction

# Load test data

dir_path = "data/test_data2/"
# Load Pickled Data
file_path = dir_path + "data_eval_dict.pickle"
with open(file_path, 'rb') as f: 
    data_dict = pickle.load(f, encoding="bytes")
    print(data_dict.keys())
testobj1 = data_dict['obj_loc']
testobj2 = data_dict['obj2_loc']
testtraj = data_dict['path']
testbuffer = data_dict['buffer']
test_env_traj = data_dict['env_path']
test_env_lim = data_dict['tray_lim']
test_klerg_lim = data_dict['klerg_lim']
print(testobj1, testobj2)


ycheck1 = [None]*len(testbuffer)
ycheck2 = [None]*len(testbuffer)

with torch.no_grad():
    xt = np.expand_dims(testbuffer[0][0], axis=0)
    yt = np.expand_dims(testbuffer[0][1], axis=0)

    xt = torch.FloatTensor(xt)#.squeeze()
    yt = torch.FloatTensor(yt)#.squeeze()

    _, _ , z_mu1, z_logvar1, _ = model(xt, yt)
    z_samples1 = model.reparameterize(z_mu1, z_logvar1)

    ind = 0
    for i in range(int(len(testbuffer)/2)):
        print(ind)
        xc = np.expand_dims(testbuffer[ind][0], axis=0)
        xc = torch.FloatTensor(xc)#.squeeze()

        y_pred = model.img_decode(torch.cat([z_samples1, xc], dim=1))
        img_pred = model.img_decoder(y_pred)
        ycheck1[ind] = img_pred
                
        ind +=1
        
    xt = np.expand_dims(testbuffer[ind][0], axis=0)
    yt = np.expand_dims(testbuffer[ind][1], axis=0)

    xt = torch.FloatTensor(xt)#.squeeze()
    yt = torch.FloatTensor(yt)#.squeeze()

    _, _ , z_mu1, z_logvar1, _ = model(xt, yt)
    z_samples1 = model.reparameterize(z_mu1, z_logvar1)

    for i in range(int(len(testbuffer)/2)):
        xc = np.expand_dims(testbuffer[ind][0], axis=0)
        xc = torch.FloatTensor(xc)#.squeeze()

        y_pred = model.img_decode(torch.cat([z_samples1, xc], dim=1))
        img_pred = model.img_decoder(y_pred)
        ycheck1[ind] = img_pred
        ind += 1
        
for i in range(len(testbuffer)):
    fig, axes = plt.subplots(1,2)#, figsize=(18, 6))
    axes[0].imshow(testbuffer[i][1].T)
    axes[1].imshow((ycheck1[i][0]).T)
    plt.show()
