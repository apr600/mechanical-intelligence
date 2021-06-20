import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Normal

import random
import pickle
import matplotlib.pyplot as plt
from IPython.display import clear_output

# ### FILE IMPORTS ###
# import kl_erg
# from robot_2d_klerg import Robot
# from kl_erg.vae_distr import VAEDistr

class VAE(nn.Module):
    # Main VAE code/setup
    def __init__(self, input_dim, output_dim, z_dim, s_dim, hidden_dim=128):
        super(VAE, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(input_dim+s_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ReLU(),
            # nn.Softmax(dim=1),
            nn.Linear(int(hidden_dim/2), z_dim * 2)
        )
        self.decode = nn.Sequential(
            nn.Linear(z_dim+s_dim, int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim+1)
        )
        self.z_dim = z_dim
        self.output_dim = output_dim
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
            var_data = (var_data/np.max(var_data))**4
            var_data = np.array(var_data).squeeze()
            return var_data
