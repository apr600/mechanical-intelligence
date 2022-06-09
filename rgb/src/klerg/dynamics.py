### PYTHON IMPORTS ###
import numpy as np
import numpy.random as npr

class PointMassDynamics(object):
    def __init__(self, x0=[0., 0.], dt=0.1):
        self.num_states = 2
        self.num_actions = 1

        self.x0 = x0
        self.xcurr = x0
        self.dt = dt

    def step(self, u=0.0):
        self.xcurr = np.array([self.xcurr[0]+self.xcurr[1]*self.dt, u])
        return self.xcurr      

class SingleIntegratorEnv(object):

    def __init__(self):

        self.num_states = 2
        self.num_actions = 1
        self.dt = 0.1

        self.A = np.array([
            [0., 1.0, 0.],
            [0., 0., 1.0]
        ])
        
    def get_lin(self, x, u):
        return self.A[:, :self.num_states], self.A[:, self.num_states:]
        
    def reset(self, state=None):

        if state is None:
            self.state = npr.uniform(-.1, .1, size=(self.num_states,))
        else:
            self.state = state.copy()

        return self.state.copy()

    def f(self, x, u):
        inputs = np.hstack((x, u))
        return np.dot(self.A, inputs)

    def step(self, u):
        self.state = self.state + self.f(self.state, u) * self.dt
        return self.state.copy()

class SingleIntegrator2dEnv(object):

    def __init__(self, dt=0.1):

        self.num_states = 4
        self.num_actions = 2
        self.dt = dt #0.1

        self.__A = np.array([
            [0.,0.,1.,0.],
            [0.,0.,0.,1.],
            [0.,0.,0.,0.],
            [0.,0.,0.,0.]
        ])

        self.__B = np.array([
            [0.,0.],
            [0.,0.],
            [1.,0.],
            [0.,1.]
        ]) 


    def fdx(self, x, u):
        ''' Linearization wrt x '''
        return self.__A.copy()

    def fdu(self, x, u):
        ''' Linearization wrt u '''
        return self.__B.copy()

    def get_lin(self, x, u):
        ''' returns both fdx and fdu '''
        return self.fdx(x, u), self.fdu(x, u)
    
    def reset(self, state=None):

        if state is None:
            self.state = npr.uniform(0., .1, size=(self.num_states,))
        else:
            self.state = state.copy()

        return self.state.copy()

    def f(self, x, u):
        ''' Continuous time dynamics '''
        return self.__A.dot(x) + self.__B.dot(u)

    def step(self, u):
        self.state =  self.state + self.f(self.state, u) * self.dt
        return self.state.copy()
