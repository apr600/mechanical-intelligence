import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal as mv_norm
from scipy.interpolate import RegularGridInterpolator
from scipy import interpolate
class TargetDist(object):
    
    def __init__(self, gridpts, griddata):
        self.griddata = griddata
        self.pts = gridpts
        
        self.f  = interpolate.interp2d(self.pts[:,0], self.pts[:,1], self.griddata, kind='cubic')#RegularGridInterpolator((self.pts[:,0], self.pts[:,1]), self.griddata)
        
    def gumbel_sample(self, n):
        pi = np.repeat(self.pi.reshape(-1,self.num_gauss), n, axis=0)
        z  = np.random.gumbel(loc=0, scale=1, size=pi.shape)
        return (np.log(pi) + z).argmax(axis=1)

    def sample(self, n=1):
        k   = np.random.multinomial(n, self.pi)
        s = []
        for i in range(len(self.pi)):
            s.append(npr.multivariate_normal(self.mu[i], self.std[i], k[i]))
        s = np.vstack(s)       
        return s
    
    def pdf(self, x):
        y = np.array([self.f(x[i,0], x[i,1]) for i in range(len(x))]).squeeze()
        return y
