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
# # Initialize VAE & VAE buffer
model = VAE(input_dim=5120,  z_dim=6,s_dim=2,  img_dim=5,
            hidden_dim=16)
optimizer = optim.Adam(model.parameters(), lr=3e-3)
num_learning_opt = 10000
batch_size = 10

# Load collected data
# Load Pickled Data
file_path1 = "data/unifklerg5/data_eval_dict.pickle"
with open(file_path1, 'rb') as f: 
    data_dict1 = pickle.load(f, encoding="bytes")
    print(data_dict1.keys())
obj1 = data_dict1['obj_loc']
traj1 = data_dict1['path']
vae_buffer = data_dict1['buffer']
env_traj1 = data_dict1['env_path']
env_lim = data_dict1['tray_lim']
klerg_lim = data_dict1['klerg_lim']

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

vae_buffer = np.array(vae_buffer)
for iter_opt in range(num_learning_opt):
    # Sample Buffer
    batch = np.array(vae_buffer)[np.random.choice(len(vae_buffer),
                batch_size, replace=False)]
    x, y = map(np.stack, zip(*batch))
    for ind in range(len(x)):
        x[ind] = np.array(ws_conversion(x[ind], env_lim, klerg_lim))
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
        print(loss.item())

        with torch.no_grad():
            sample = vae_buffer[np.random.choice(vae_buffer.shape[0], 1, replace=False)]
            x, y = map(np.stack, zip(*sample))
            for ind in range(len(x)):
                x[ind] = np.array(ws_conversion(x[ind], env_lim,
                                                klerg_lim))

                x = torch.FloatTensor(x)#.squeeze()
                y = torch.FloatTensor(y)#.squeeze()

                x_obj = torch.FloatTensor(np.expand_dims([obj1[0],
                                        obj1[2]],axis=0))

                _,_, z_mu, z_logvar, z_samples = model(x, y)
                z_samples = model.reparameterize(z_mu, z_logvar)

            var_list = []
            for i in range(len(pts)):
                x = torch.FloatTensor(np.expand_dims(pts[i],
                                                 axis=0))
                img_logvar = model.imgvar_decoder(torch.cat([z_samples, x],
                                                     dim=1))
                # img_logvar = torch.clamp(img_logvar, -5, 2)
                var_list.append(img_logvar[0].exp().detach().numpy())
        var_list = np.array(var_list)
        plt.contourf(xx,yy,var_list.reshape(xx.shape))
        plt.plot(obj_ss[0], obj_ss[1],'k*')
        plt.pause(1)


# # Reconstruct img prediction

sample = vae_buffer[np.random.choice(vae_buffer.shape[0], 1, replace=False)]

x, y = map(np.stack, zip(*sample))
for ind in range(len(x)):
    x[ind] = np.array(ws_conversion(x[ind], env_lim, klerg_lim))

x = torch.FloatTensor(x)#.squeeze()
y = torch.FloatTensor(y)#.squeeze()

x_obj = torch.FloatTensor(np.expand_dims([obj1[0],obj1[2]],axis=0))

with torch.no_grad():
    img_pred, img_logvar , z_mu, z_logvar, z_samples = model(x, y)
    z_samples = model.reparameterize(z_mu, z_logvar)

    print(z_samples.shape, x.shape)
    y_pred = model.img_decode(torch.cat([z_samples, x], dim=1))
    img_pred = model.img_decoder(y_pred)

    print(z_samples.shape, x_obj.shape)
    yo_pred = model.img_decode(torch.cat([z_samples, x_obj], dim=1))
    imgo_pred = model.img_decoder(yo_pred)

img_new = img_pred.detach().numpy()[0].T
img_new = np.clip(img_new,0,1)

imgo_new = imgo_pred.detach().numpy()[0].T
imgo_new = np.clip(imgo_new,0,1)

img_orig = y.detach().numpy()[0].T

plt.imshow(img_orig)
plt.show()

plt.imshow(img_new)
plt.show()

plt.imshow(imgo_new)
plt.show()

lim = np.array([[-0.2,0.2],[-0.3,-0.7]])
xlist = np.linspace(klerg_lim[0,0],klerg_lim[0,1],21)
ylist = np.linspace(klerg_lim[1,0],klerg_lim[1,1],21)

xx, yy = np.meshgrid(xlist, ylist)
pts = np.array([xx.ravel(), yy.ravel()]).T
var_list = []
with torch.no_grad():
    for i in range(len(pts)):
        x = torch.FloatTensor(np.expand_dims(pts[i],
                                             axis=0))
        img_logvar = model.imgvar_decoder(torch.cat([z_samples, x],
                                                 dim=1))
        # img_logvar = torch.clamp(img_logvar, -5, 2)
        var_list.append(img_logvar[0].exp().detach().numpy())
var_list = np.array(var_list)
print(var_list.shape)

plt.contourf(xx,yy,var_list.reshape(xx.shape))
# plt.plot(obj_ss[0], obj_ss[1],'k*')
plt.show()
