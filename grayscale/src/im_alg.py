import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import math

from klerg.target_dist import TargetDistr
from klerg.dynamics import SingleIntegrator2dEnv
class IM():
    # Implement Information Maximizing
    def __init__(self, x0, wslim,ts, target_dist=None):
        # Set up Workspace
        self.lim = wslim

        # Set up Target Distribution
        if target_dist is None: 
            self.target_dist = TargetDistr()
            print("Gaussian Peaks: {}".format(self.target_dist.mu))
        else: 
            self.target_dist=target_dist
            print("Using input dist as target")

        # Plot Target Dist
        numpts = 50
        x, y = np.meshgrid(np.linspace(self.lim[0],self.lim[1],numpts), np.linspace(self.lim[0],self.lim[1],numpts))
        samples = np.c_[x.ravel(), y.ravel()]
        data = self.target_dist.pdf(samples)
        plt.contourf(x,y, np.reshape(data, (numpts, numpts)))
        plt.colorbar()
        plt.show()

        # Set up robot
        self.robot = SingleIntegrator2dEnv(dt=ts)
        self.robot.reset(np.array(x0))
    
        

    def step(self, num_target_samples=50, max_dist=0.2):
        current_pos = self.robot.state.copy()[:2]
        sample_lim = [current_pos - max_dist, current_pos + max_dist]  
        samples = npr.uniform(low=sample_lim[0], high=sample_lim[1], size=(num_target_samples, len(current_pos)))
        p = self.target_dist.pdf(samples).squeeze()+1e-3

        max_i = np.argmax(p)
        xnew = samples[max_i]

        self.robot.reset(xnew)
        return xnew




if __name__ == "__main__":

    x0 = [npr.uniform(-1,1), npr.uniform(-1,1), 0,0]
    robot = IM(x0=x0, wslim=[-1.,1.])
    path = []
    num_steps = 300
    for i in range(num_steps): 
        # state, _ = robot.step() # default for input target dist
        state = robot.step(num_target_samples= 25, max_dist=0.2) # Optimized for uniform
        path.append(state)
    path = np.array(path)

    numpts = 50
    pts = np.linspace(robot.lim[0],robot.lim[1],numpts)
    x, y = np.meshgrid(pts, pts)
    gridpts = np.array([np.ravel(x), np.ravel(y)]).T
    plt.imshow(robot.target_dist.pdf(gridpts).reshape((numpts, numpts)), origin="lower", extent=[-1,1,-1,1])

    plt.plot(path[0,0], path[0,1], 'r*')
    plt.plot(path[:,0], path[:,1], 'k.')
    plt.show()

        

        
