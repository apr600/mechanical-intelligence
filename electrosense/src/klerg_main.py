import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt


from dynamics import SingleIntegrator2dEnv
from memory_buffer import MemoryBuffer
from target_dist import TargetDistr
from klerg_utils import *
from barrier import BarrierFunction
class Robot():
    """ Robot class that runs the KL-Erg MPC Planner """
    def __init__(self, x0, wslim, explr_idx,  target_dist=None, ts=0.1, horizon=10, buffer_capacity=100, std=0.05):

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
        print("Close Plot to continue running algorithm")
        plt.contourf(x,y, np.reshape(data, (numpts, numpts)))
        plt.show(block=False)
        
        
        # Set up robot
        self.robot = SingleIntegrator2dEnv(dt=ts)
        self.robot.reset(np.array(x0))
        

        # Set up KL-Erg Planner
        self.planner = SingleIntegrator2dEnv(dt=ts)
        self.planner.reset(np.array(x0))
        
        self.horizon = horizon
        self.explr_idx = explr_idx
        self.std = std
        self.u = [np.zeros(self.planner.num_actions) for i in range(self.horizon)]
        self.memory_buffer = MemoryBuffer(buffer_capacity)

        self.barrier = BarrierFunction(b_lim = [self.lim]*2)

    def step(self, num_target_samples= 50, num_traj_samples=30, R=0.01, alpha=1e-3): 

        self.kldiv_planner(num_target_samples= num_target_samples, num_traj_samples= num_traj_samples, R=R, alpha=alpha)
        self.robot.step(self.u[0])
        self.memory_buffer.push(self.robot.state.copy())
        return self.robot.state, self.u[0]
        
    def kldiv_planner(self, num_target_samples, num_traj_samples, R=1, alpha=1e-3):

        # Make sure planner state is the same as robot state
        self.planner.reset(self.robot.state.copy())
        
        # Update u list (move each u up in time by 1)
        self.u[:-1] = self.u[1:]
        self.u[-1] = np.zeros(self.planner.num_actions)

        grad_list = []
        traj_list = []

        # Sample Target Distribution
        samples = npr.uniform(self.lim[0], self.lim[1],size=(num_target_samples, len(self.explr_idx)))
        p = self.target_dist.pdf(samples).squeeze()+1e-3
        p /= np.sum(p)
        p = np.log(p)
        p -= np.max(p)
        p = np.exp(p)

        # Forward Pass
        for i in range(self.horizon):
            # Calculate derivatives
            A, B = self.planner.get_lin(self.planner.state.copy(), self.u[i])
            # barr_grad = grad(barrier,self.explr_idx, self.lim )
            # dbarrdx = barr_grad(self.planner.state.copy())
            dbarrdx = self.barrier.dbarr(self.planner.state.copy())
            grad_list.append((A,B, dbarrdx))
            # Step x state forward
            self.planner.step(self.u[i])
            traj_list.append(self.planner.state.copy())

        # Sample Trajectory Distribution
        if len(self.memory_buffer) > num_traj_samples:
            traj_states = self.memory_buffer.sample(num_traj_samples)
        else:
            traj_states = self.memory_buffer.sample(len(self.memory_buffer))
        traj_samples = np.stack(traj_list + traj_states)
        q = np.array([traj_footprint(traj_samples, s, self.explr_idx, self.std) for s in samples])+1e-3
        norm_factor = np.sum(q)
        q = q/norm_factor
        q = np.log(q)
        q -= np.max(q)
        q=np.exp(q)
        # Backwards pass
        rho = np.zeros(self.planner.state.shape)
        importance_ratio = p/(q+1e-1)

        for i in reversed(range(self.horizon)):
            A, B, dbarrdx = grad_list[i]
            x = traj_list[i]
            # kldiv_grad = grad(state_footprint)
            dgdx = np.sum([qi*kldiv_grad(x, s, self.explr_idx, self.std) for s, qi in zip(samples, importance_ratio)], axis=0)
            rhodot = dgdx-A.T.dot(rho)-dbarrdx
            rho = rho - rhodot*self.planner.dt
            # du = R*self.u[i] + B.T.dot(rho)
            # self.u[i] =np.clip(self.u[i] - alpha*du,-2,2)
            self.u[i] = np.clip(-(1/R)*B.T.dot(rho), -2,2)#TODO: clip to -2,2


if __name__ == "__main__":

    x0 = [npr.uniform(-1,1), npr.uniform(-1,1), 0,0]
    # robot = Robot(x0=x0, wslim=[-1.,1.], ts=0.1, explr_idx=[0,1], buffer_capacity=10000) # for entropy sampling (and unif1)
    robot = Robot(x0=x0, wslim=[-1.,1.], ts=0.1, explr_idx=[0,1], buffer_capacity=10000)
    path = []
    num_steps = 300
    for i in range(num_steps): 
        # state, _ = robot.step() # default is what is used for entropy sampling (and unif 1)
        state, _ = robot.step(num_target_samples= 100, num_traj_samples=num_steps, R=0.01, alpha=1e-3)
        path.append(state)
    path = np.array(path)

    numpts = 50
    pts = np.linspace(robot.lim[0],robot.lim[1],numpts)
    x, y = np.meshgrid(pts, pts)
    gridpts = np.array([np.ravel(x), np.ravel(y)]).T
    plt.imshow(robot.target_dist.pdf(gridpts).reshape((numpts, numpts)), origin="lower", extent=[-1,1,-1,1])

    plt.plot(path[0,0], path[0,1], 'r*')
    plt.plot(path[:,0], path[:,1], 'k.')
    # plt.xlim(robot.lim[0], robot.lim[1])
    # plt.ylim(robot.lim[0], robot.lim[1])
    plt.show()
