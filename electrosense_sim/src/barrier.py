import numpy as np

class BarrierFunction(object):

    def __init__(self, b_lim, ergodic_dim=2, barr_weight=100.0, b_buff=0.01):
        ''' Barrier Function '''
        self.ergodic_dim = ergodic_dim
        self.b_lim = np.copy(b_lim)
        self.barr_weight = barr_weight
        for i in range(self.ergodic_dim):
            self.b_lim[i][0] = self.b_lim[i][0] + b_buff
            self.b_lim[i][1] = self.b_lim[i][1] - b_buff
        print(self.b_lim)

    def barr(self,x):
        barr_temp = 0.
        for i in range(self.ergodic_dim):
            if x[i] >= self.b_lim[i][1]:
                barr_temp += self.barr_weight*(x[i] - (self.b_lim[i][1]))**2
            elif x[i] <=  self.b_lim[i][0]:
                barr_temp += self.barr_weight*(x[i] - (self.b_lim[i][0]))**2
        return barr_temp

    def dbarr(self, x):
        dbarr_temp = np.zeros(len(x))
        for i in range(self.ergodic_dim):
            if x[i] >= self.b_lim[i][1]:
                dbarr_temp[i] += 2*self.barr_weight*(x[i] - (self.b_lim[i][1]))
            elif x[i] <=  self.b_lim[i][0]:
                dbarr_temp[i] += 2*self.barr_weight*(x[i] - (self.b_lim[i][0]))
        return dbarr_temp
