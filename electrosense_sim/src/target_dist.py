import numpy as np
import numpy.random as npr
from scipy.stats import norm
from scipy.stats import multivariate_normal as mv_norm
class TargetDistr(object):
    def __init__(self):
        # self.mu = [ np.array([0.2, 0.2])]
        self.mu = [np.array([-0.3, 0.3]), np.array([0.7, 0.3])]
        self.pi = [.5]*len(self.mu)
        self.std = [np.eye(2)*0.01]*len(self.mu)
        self.num_gauss = len(self.pi)        
        print("Number of Gaussian Peaks: {}, Peaks: {}".format(self.num_gauss, self.mu))
    
    def init_uniform_grid(self, x):
        assert len(x.shape) > 1, 'Input needs to be a of size N x n'
        # assert x.shape[1] == self.explr_dim, 'Does not have right exploration dim'
        val = np.ones(x.shape[0])
        val /= np.sum(val)
        val += 1e-5
        return val

    def sample(self, n=1):
        k   = np.random.multinomial(n, self.pi)
        s = []
        for i in range(len(self.pi)):
            s.append(npr.multivariate_normal(self.mu[i], self.std[i], k[i]))
        s = np.vstack(s)       
        return s

    def pdf(self, x):
        return self.init_uniform_grid(x)
        # out = 0.
        # for mu, pi, std in zip(self.mu, self.pi, self.std):
        #     out += pi * mv_norm.pdf(x, mu, std)
        # return out + 1e-4
