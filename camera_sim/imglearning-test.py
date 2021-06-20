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


# # Initialize VAE & VAE buffer
model = VAE(input_dim=5120,  z_dim=6,s_dim=2,  img_dim=5, hidden_dim=128)
optimizer = optim.Adam(model.parameters(), lr=3e-3)
num_learning_opt = 2000
batch_size = 10

# Load collected data
# Load Pickled Data
file_path1 = "data/misc/data_dict.pickle"
with open(file_path1, 'rb') as f: 
    data_dict1 = pickle.load(f, encoding="bytes")
    print(data_dict1.keys())
obj1 = data_dict1['obj_loc']
traj1 = data_dict1['path']
vae_buffer = data_dict1['buffer']
env_traj1 = data_dict1['env_path']



sample = vae_buffer[0]
img = cv2.imread("/home/anon/ahalya/LearningSensoryStructure/franka-sim/data/misc/21_img.png")
print(img.shape, sample[1].shape)
x = np.expand_dims(sample[0], axis=0)
img = img[::3, ::3, :]
y = np.expand_dims((img/255.0).T, axis=0)


for iter_opt in range(num_learning_opt):
    # Sample Buffer
    x = torch.FloatTensor(x)#.squeeze()
    y = torch.FloatTensor(y)#.squeeze()
    # Run optimization
    y_pred, img_logvar_scalar, z_mu, z_logvar, z_samples = model(x, y)
    # print(img_logvar_scalar.shape, img_logvar_scalar[0].item())
    img_logvar = img_logvar_scalar.expand_as(y_pred)

    p = Normal(z_mu, z_logvar.exp())
    q = Normal(y_pred, img_logvar.exp())

    loss = - torch.mean(q.log_prob(y)) \
                - .1*  torch.mean(0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp()).sum(1)) #0.01
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    if iter_opt %100 ==0:
        print(loss.item())

# Reconstruct img prediction

# xr = torch.FloatTensor(xr).unsqueeze(axis=0)
# yr = torch.FloatTensor(yr).unsqueeze(axis=0)

# with torch.no_grad():
#     img_pred, img_logvar , z_mu, z_logvar, z_samples = model(x, y)
#     z_samples = vae.reparameterize(z_mu, z_logvar)
#     y_pred = vae.img_decode(torch.cat([z_samples, x], dim=1))
#     img_pred = vae.img_decoder(y_pred)
img_new = y_pred.detach().numpy()[0].T
img_new = np.clip(img_new,0,1)
img_orig = y.detach().numpy()[0].T

plt.imshow(img)
plt.show()

plt.imshow(img_new)
plt.show()
