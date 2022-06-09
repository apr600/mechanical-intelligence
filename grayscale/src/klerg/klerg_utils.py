#!usr/bin/env python

### Python Imports ###
# import autograd.numpy        as np
# import autograd.numpy.random as npr 

# from autograd.scipy.stats import norm
# from autograd             import jacobian, grad

import numpy as np
import numpy.random as npr
from scipy.stats import norm

def state_footprint(x, s, explr_idx, std):
    """state footprint for single point"""
    return np.exp(-0.5 * np.sum(np.square(x[explr_idx] - s)/std))

def kldiv_grad(x,s, explr_idx, std):
    """ gradient of state footprint """
    grad = np.zeros(x.shape)
    grad[explr_idx] = -0.5*2*((x[explr_idx]-s)/std)*np.exp(-0.5* np.sum(np.square(x[explr_idx] - s)/std))
    return grad

def traj_footprint(traj, s, explr_idx, std):
    """ time-averaged footprint of trajectory (used to calc q)"""
    pdf = -0.5 * np.sum(np.square(traj[:,explr_idx]-s)/std, 1)
    pdf = np.mean(np.exp(pdf), 0)+1e-5
    return pdf

def barrier(x, explr_idx, lim):
    barr_cost = np.sum(np.tanh(10*(x[explr_idx]-(np.array([lim[1]]*len(explr_idx))))) * np.tanh(10.*(x[explr_idx]-(np.array([lim[0]]*len(explr_idx))))))
    return barr_cost

def backward(A, B, dbarrdx, dgdx, x, u, rho, dt, R, alpha):
    rhodot = dgdx-A.T.dot(rho)-dbarrdx
    rho = rho- rhodot*dt
    du = R*u+B.T.dot(rho)
    unew = u-alpha*du
    return unew
