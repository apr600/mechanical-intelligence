import numpy as np
import numpy.random as npr

class MeasurementModel(object):
    def __init__(self):
        self.xx, self.yy = np.meshgrid(np.linspace(-0.5,0.5,21), 
                                       np.linspace(-0.5,0.5,21))
        self.samples = np.c_[self.xx.ravel(), self.yy.ravel()]
        self.n = self.samples.shape[0]
    def __call__(self, offset=np.array([[0.,0.]])):
        x = self.samples
        y = 5*np.exp(-10.0 * np.linalg.norm(x-offset, axis=1)) *\
            np.tanh(5.0 * (x[:,0]-offset[:,0])) * \
            np.tanh(5.0 * (x[:,1]-offset[:,1])) 
        return (y + np.random.normal(0.,0.01, size=y.shape)).reshape(1,self.n)

class MeasurementModel_nonoise(object):
    def __init__(self):
        self.xx, self.yy = np.meshgrid(np.linspace(-0.5,0.5,21), 
                                       np.linspace(-0.5,0.5,21))
        self.samples = np.c_[self.xx.ravel(), self.yy.ravel()]
        self.n = self.samples.shape[0]
    def __call__(self, offset=np.array([[0.,0.]])):
        x = self.samples
        y = 5*np.exp(-10.0 * np.linalg.norm(x-offset, axis=1)) *\
            np.tanh(5.0 * (x[:,0]-offset[:,0])) * \
            np.tanh(5.0 * (x[:,1]-offset[:,1])) 
        return (y).reshape(1,self.n)
    
