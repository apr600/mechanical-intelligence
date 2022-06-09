import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
# import cv2

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Normal

import torchvision
from torchvision import transforms
from torchvision.utils import save_image

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=5):
        return input.view(input.size(0), size, 32, 32)

class Sin(nn.Module):
    def forward(self, input):
        # w = torch.rand(input.shape[0], input.shape[1])*(np.sqrt(6)/input.shape[0])
        w = .1
        return torch.sin(w*input)

class VAE(nn.Module):
    # Image VAE code/setup
    def __init__(self, input_dim, z_dim, s_dim, img_dim, hidden_dim=128):
        # input/output dim=camimg_dim, z_dim=# of latent space, s_dim=dim of conditional [x,y]
        super(VAE, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(input_dim+s_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), int(hidden_dim/4)),
            nn.ReLU(),
            # nn.Softmax(dim=1),
            nn.Linear(int(hidden_dim/4), z_dim * 2)
        )
        self.decode = nn.Sequential(
            nn.Linear(z_dim+s_dim, int(hidden_dim/4)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/4), int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), hidden_dim),
            nn.ReLU(),
            # nn.Softmax(dim=1),
            nn.Linear(hidden_dim, input_dim+1)
        )
        self.z_dim = z_dim
        self.output_dim = input_dim
        self.init=False

    def reparameterize(self, mu, logvar):
        var = logvar.exp()
        eps = torch.randn_like(var.detach())
        z = mu + eps*var
        return z

    def forward(self, x, y):
        z_out = self.encode(torch.cat([y, x], dim=1))
        z_mu, z_logvar = z_out[:,:self.z_dim], z_out[:, self.z_dim:]
        z_logvar = torch.clamp(z_logvar, -5, 2)
        z_samples = self.reparameterize(z_mu, z_logvar)
        y_out = self.decode(torch.cat([z_samples, x], dim=1))
        y_pred, y_logvar = y_out[:, 1:], y_out[:, 0].unsqueeze(1)
        return  y_pred, torch.clamp(y_logvar, -5, 2), z_mu, z_logvar, z_samples

    # Functions for Target Dist
    
    def init_uniform_grid(self, x):
        assert len(x.shape) > 1, 'Input needs to be a of size N x n'
        assert x.shape[1] == 2, 'Does not have right exploration dim'

        val = np.ones(x.shape[0])
        val /= np.sum(val)
        val += 1e-5
        return val

    def update_dist(self, xr, y):
        with torch.no_grad():
            self.y_pred, self.y_logvar, self.z_mu, self.z_logvar, self.z_samples = self.forward(xr, y)
        self.init = True

    def pdf(self, samples=None):
        if not self.init:
            return self.init_uniform_grid(samples)
        else: 
            var_data = [None]*len(samples)
            with torch.no_grad():
                for i in range(len(samples)):
                    xr = np.expand_dims(samples[i], 0)
                    xr = torch.FloatTensor(xr)
                    y_out = self.decode(torch.cat([self.z_samples, xr], dim=1))
                    y_pred, y_logvar = y_out[:, 1:], y_out[:, 0]
                    var_data[i] = y_logvar.exp().detach().numpy()
            var_data = (var_data/np.max(var_data))
            var_data = np.array(var_data).squeeze()
            return var_data

 
if __name__ == "__main__":
    indim = 38*38
    print(indim)
    vae = VAE(input_dim=indim,  z_dim=6,s_dim=2,  img_dim=5, hidden_dim=16)
    img = plt.imread("/home/anon/ahalya/LearningSensoryStructure/franka-sim/data/misc/21_img.png")
    img = img[::3,::3,:]
    img = img[::2,::2,:]
    print(img.shape)
    # cv2.imshow('image', img)
    # cv2.waitKey(1000)


    # # img = img.T
    # # print(img.shape)
    # img = np.mean(img, axis=2)
    # print(img.shape)
    # cv2.imshow('image', img)
    # cv2.waitKey(1000)
    
    y = torch.FloatTensor(img.flatten()).unsqueeze(0)
    x = [0.2,0.2]
    x = torch.FloatTensor(x).unsqueeze(0)
    print(y.shape, x.shape)
    z_out  = vae.encode(torch.cat([y, x], dim=1))
    print("z shape", z_out.shape)
    z_mu, z_logvar = z_out[:,:vae.z_dim], z_out[:, vae.z_dim:]
    z_logvar = torch.clamp(z_logvar, -5, 2)
    z_samples = vae.reparameterize(z_mu, z_logvar)
    # y_pred = vae.img_decode(torch.cat([z_samples, x], dim=1))
    # print("y_pred shape: ", y_pred.shape)
    y_out = vae.decode(torch.cat([z_samples, x], dim=1))
    y_pred = y_out[:,1:]
    y_logvar = y_out[:,0].unsqueeze(axis=1)
    # print(img_pred.shape, img_logvar.shape, img_logvar[:,0], img_logvar[0,0], img_logvar[0,0].exp().detach().numpy())

    img_new = y_pred.detach().numpy().reshape((38,38))
    np.clip(img_new, 0, 1)

    plt.imshow( img_new)
    plt.show()
