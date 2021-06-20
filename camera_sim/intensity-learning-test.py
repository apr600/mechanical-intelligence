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
from vae.vae_intensity import VAE
from vae.vae_buffer import ReplayBuffer
from franka.franka_utils import *


# env_lim = np.array([[-0.2,0.2],[-0.3,-0.7]])
# klerg_lim = np.array([[-1., 1.],[-1.,1.]])
in_dim = 75*75


# Load collected data
# Load Pickled Data
# file_path1 = "data/unifklerg5/data_eval_dict.pickle"
file_path1 = "data/intensity/entklerg2/data_eval_dict.pickle"

with open(file_path1, 'rb') as f: 
    data_dict1 = pickle.load(f, encoding="bytes")
    print(data_dict1.keys())
obj1 = data_dict1['obj_loc']
traj1 = data_dict1['path']
vae_buffer = data_dict1['buffer']
env_traj1 = data_dict1['env_path']
env_lim = data_dict1['tray_lim']
klerg_lim = data_dict1['klerg_lim']


# # Initialize VAE & VAE buffer
model = VAE(input_dim=1444,  z_dim=16, s_dim=2,  img_dim=5,
            hidden_dim=150)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_learning_opt = 16000
batch_size = 64#len(vae_buffer)#64


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
sample = vae_buffer[0]
# xs = np.expand_dims(sample[0], axis=0)
# ys = np.expand_dims(sample[1], axis=0)

for iter_opt in range(num_learning_opt):
    # Sample Buffer
    batch = np.array(vae_buffer)[np.random.choice(len(vae_buffer),
                batch_size, replace=False)]
    # batch = np.array(vae_buffer)
    xs, ys = map(np.stack, zip(*batch))
    xt = xs
    
    # ys = ys[:,:, ::2, ::2]
    # yt = np.mean(ys, axis=1).reshape((batch_size, 1444))
    yt=ys
        
    x = torch.FloatTensor(xt).squeeze()
    y = torch.FloatTensor(yt).squeeze()

    # Run optimization
    y_pred, img_logvar_scalar, z_mu, z_logvar, z_samples = model(x,y)
    ysize = y_pred.size()
    # img_logvar = img_logvar_scalar.unsqueeze(2).unsqueeze(3).expand(ysize)

    img_logvar = img_logvar_scalar.expand(ysize)

    p = Normal(z_mu, z_logvar.exp())
    q = Normal(y_pred, img_logvar.exp())

    loss = - torch.mean(q.log_prob(y)) \
                - .01*  torch.mean(0.5 * (1 + z_logvar - z_mu.pow(2)
                - z_logvar.exp()).sum(1)) #0.01
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    if iter_opt %100 ==0:
        print(loss.item())

        # with torch.no_grad():
        #     sample = vae_buffer[np.random.choice(vae_buffer.shape[0], 1, replace=False)]
        #     x, y = map(np.stack, zip(*sample))
        #     yt = [None]
        #     for ind in range(len(x)):
        #         x[ind] = np.array(ws_conversion(x[ind], env_lim,
        #                                         klerg_lim))
        #         yt[ind] = np.mean(y[ind], axis=0).flatten()

        #     x = torch.FloatTensor(x)#.squeeze()
        #     y = torch.FloatTensor(yt)#.squeeze()

        #     x_obj = torch.FloatTensor(np.expand_dims([obj1[0],
        #                                 obj1[2]],axis=0))

        #     _,_, z_mu, z_logvar, z_samples = model(x, y)
        #     z_samples = model.reparameterize(z_mu, z_logvar)

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


torch.save(model, "model_int_check.pth")

# torch.save({
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#             }, "model_ent_state.tar")
torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, "model_int_state1.pth")
