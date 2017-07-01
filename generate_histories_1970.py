import numpy as np
import random as rnd
import pandas as pd
from python.model import Model
from python.parameters import CRITERION, MAXITERATIONS
from python.constants import BETA, R, IOTA, TAU, pp_range, o_range, z_shock_range
from python.distributions import PPDDISTRIBUTION, ZDISTRIBUTION, PIMATRIX


PERIODS = 6                               # Number of periods per one epoch
EPOCHS = 6                               # Number of epochs  
AGENTS = 7000                            # Number of agents


def rouwen(rho, mu, step, num):
    '''
    INPUTS:
    rho  - persistence (close to one)
    mu   - mean and the middle point of the discrete state space
    step - step size of the even-spaced grid
    num  - number of grid points on the discretized process

    OUTPUT:
    dscSp  - discrete state space (num by 1 vector)
    transP - transition probability matrix over the grid
    '''

    # discrete state space
    dscSp = np.linspace(mu -(num-1)/2*step, mu +(num-1)/2*step, num).T

    # transition probability matrix
    q = p = (rho + 1)/2.
    transP = np.array([[p**2, p*(1-q), (1-q)**2], \
                    [2*p*(1-p), p*q+(1-p)*(1-q), 2*q*(1-q)], \
                    [(1-p)**2, (1-p)*q, q**2]]).T

    while transP.shape[0] <= num - 1:

        # see Rouwenhorst 1995
        len_P = transP.shape[0]
        transP = p*np.vstack((np.hstack((transP,np.zeros((len_P,1)))), np.zeros((1, len_P+1)))) \
        + (1-p)*np.vstack((np.hstack((np.zeros((len_P, 1)), transP)), np.zeros((1, len_P+1)))) \
        + (1-q)*np.vstack((np.zeros((1, len_P+1)), np.hstack((transP, np.zeros((len_P, 1)))))) \
        + q*np.vstack((np.zeros((1, len_P+1)), np.hstack((np.zeros((len_P, 1)), transP))))

        transP[1:-1] /= 2.

    # ensure columns sum to 1
    if np.max(np.abs(np.sum(transP, axis=1) - np.ones(transP.shape))) >= 1e-12:
        print('Problem in rouwen routine!')
        return None
    else:
        return transP.T, dscSp
        

class NewModel(Model):
    def __init__(self):
        self.bound = [0.2, 0.2, 0.1]
        self.AUGMA = 22000
        self.AUGMH = 4320
        
        (self.AMIN, self.AMAX, self.NBA) = (25000, 525000, 50)
        (self.HMIN, self.HMAX, self.NBH) = (8900, 189000, 50)
        
        self.JOBS = 9
        self.CRITERION = CRITERION
        self.MAXITERATIONS = MAXITERATIONS
        self.BETA = BETA
        self.R =  R
        self.IOTA = IOTA
        self.TAU = TAU
        self.ZSHOCKS = 3
        self.pp_range = pp_range
        self.o_range = o_range
        self.PIMATRIX, self.z_shock_range = rouwen(0.9, 0, 1/3, 3)
                
        self.PSI = 0.4
        self.CHI = 0.04
        self.WAGECONSTANT = 2.24
        self.ALPHA = 0.0064
        self.ZETA= [0, -0.31, -0.57, -0.28, -0.41, -0.55, -0.35, -0.63, -0.80]
        self.GAMMA = [0.71, 0.25, 0.089]
        self.XI = -0.05
        self.updade_parameters()
           
    def updade_parameters(self):
        if self.NBA > 1:
            self.DELTAA = (self.AMAX - self.AMIN) / (self.NBA - 1)     
        else:
            self.DELTAA = 1
        if self.NBH > 1:
            self.DELTAH = (self.HMAX - self.HMIN) / (self.NBH - 1)     
        else:
            self.DELTAH = 1

        self.a_grid = np.linspace(self.AMIN, self.AMAX, self.NBA)
        self.h_grid = np.linspace(self.HMIN, self.HMAX, self.NBH)
        
        self._size = len(self.pp_range)*len(self.o_range)*self.JOBS
        self._full_size = self.NBA*self.NBH

        dr = np.ones( (self._full_size, self.ZSHOCKS), dtype=int )
        v = np.zeros( (self._full_size, self.ZSHOCKS), dtype=float )
        self.v = [v] * ( self._size )
        self.dr = [dr] * ( self._size ) 


    def grid_wage(self, j, pp, h_prime, h, o, z_shock):
        '''
        Earnings function on grid
        
        :param int j: current job type
        :param int pp: 1 = receiving PP, 0 = not receiving PP
        :param int h_prime: next period cumulative hours worked
        :param int h: current peruod cumulative hours worked
        :param bool o: True = changed job, False = kept job
        :param int z_shock: z-shock index
        '''
        if h > self.HMAX - self.bound[0]*self.AUGMH:
            _h = (self.bound[1] + self.bound[2]*pp)*self.AUGMH
        else:
            _h = h_prime - h
        return self.wage(j, pp, _h, h, o, self.z_shock_range[z_shock])
    
    def consumption(self, a_prime, a, w):
        '''
        Consumption
        
        :param float a_prime: next period assets
        :param float a: current peruod assets
        :param float w: wage
        '''
        return (1 + self.R)*a - a_prime + (1 - self.TAU)*w
    
    def utility_matrix(self, index):
        (pp, j, o) = self.map_from_index(index)
        _temp_um = np.zeros((self._full_size, self._full_size, 
                                         self.ZSHOCKS), dtype=float)
        _temp_um.fill(np.nan) 
        for z_shock in range(self.ZSHOCKS): 
            for h_start in self.h_grid:
                for a_start in self.a_grid:
                    for h_end in self.h_grid:
                        if h_end < h_start or h_end - h_start > self.AUGMH:
                            continue
                        for a_end in self.a_grid:
                            if a_start - a_end > self.AUGMA:
                                continue
                            _c = self.grid_consumption(j, pp, h_end, h_start, 
                                                 a_end, a_start, o, z_shock);
                            if _c <= 0:
                                continue
                            if h_start > self.HMAX - self.bound[0]*self.AUGMH:
                                _l = self.bound[1] + self.bound[2]*pp
                            else:
                                _l = self.grid_labor(h_end, h_start)
                            _idx_1 = self.map_from_grid(a_start, h_start)
                            _idx_2 = self.map_from_grid(a_end, h_end)
                            _temp_um[_idx_1, _idx_2, z_shock] = \
                                     self.utility(_c, _l)
        print("Matrix: {0:2d}; Utility matrix calculated.".format(index))
        return _temp_um
    
    def utility(self, c, l):
        '''
        Utility function on grid

        :param float c: consumption on grid
        :param float l: labor supply on grid
        '''
        _c = c
        return (_c**(1-self.IOTA))/(1-self.IOTA) - \
               self.CHI*(l**(1+self.PSI))/(1+self.PSI)

def simulate_agent(m, agent, epoch):
    _dr_idx = rnd.randint(0, 1)
    _h = m.h_grid[_dr_idx // m.NBA]
    _a = m.a_grid[_dr_idx % m.NBA]
    res = np.array( [(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]*PERIODS )
    _chosen = -1
    for period in range(PERIODS):    
        _per = 0
        _temp_v = [0]*m.JOBS
        _temp_idx = [0]*m.JOBS
        _temp_state = [(0,0,0)]*m.JOBS
        for j in range(m.JOBS):
            _pp_prob = rnd.random()
            if (_pp_prob < PPDDISTRIBUTION[j][_per]):
                pp = 1
            else:
                pp = 0
            _z_shock_prob = rnd.random()
            _z_index = 0
            _sum = ZDISTRIBUTION[_z_index]
            while (_sum < _z_shock_prob):
                _z_index += 1
                _sum += ZDISTRIBUTION[_z_index]
            _temp_idx[j] = m.dr[m.map_to_index(pp, j, 
                j == _chosen)].reshape(m.NBA*m.NBH, m.ZSHOCKS)[_dr_idx, _z_index]
            _temp_v[j] = m.v[m.map_to_index(pp, j, 
                j == _chosen)].reshape(m.NBA*m.NBH, m.ZSHOCKS)[_temp_idx[j], _z_index]
            _temp_state[j] = (pp, j == _chosen, m.z_shock_range[_z_index])
        _chosen = np.argmax(_temp_v)
        _l =  m.labor(m.h_grid[_temp_idx[_chosen] // m.NBA], 
                    m.h_grid[_dr_idx // m.NBA]) / m.AUGMH

        _w = m.wage( _chosen, 
                    _temp_state[_chosen][0], 
                    _l*m.AUGMH, 
                    m.h_grid[_dr_idx // m.NBA], 
                    _temp_state[_chosen][1],
                    _temp_state[_chosen][2] )
        if _w == 0:
            _lw = 0
        else:
            _lw = np.log(_w)
        
        _c = m.consumption(m.a_grid[_temp_idx[_chosen] % m.NBA], 
                    m.a_grid[_dr_idx % m.NBA], _w)
        _dr_idx = _temp_idx[_chosen]
        _h = m.h_grid[_dr_idx // m.NBA]
        _a = m.a_grid[_dr_idx % m.NBA]
        
        res[period] = (agent, epoch*(PERIODS-1) + period + 25, 
                    _chosen + 1, _temp_state[_chosen][0], 
                    np.log(_h), 
                    _l, _c, _lw, 
                    _temp_state[_chosen][2], _a)
    return res
        
if __name__ == '__main__':
    _h = 8900
    _a = 22000
    for epoch in range(EPOCHS):
        m = NewModel()
        m.NBH = 3*PERIODS
        m.HMIN = _h
        m.HMAX = m.HMIN + m.AUGMH*PERIODS
        m.NBA = 2*PERIODS
        m.AMIN = _a
        m.AMAX = m.AMIN + 300000
        print("Epoch: ", epoch)
        print("[", m.HMIN, m.HMAX, "] [", m.AMIN, m.AMAX, "]")
        m.updade_parameters()
        m.evaluate_model()
        _a = m.AMAX
        _h = m.HMAX
        for agent in range(1, AGENTS + 1):
            res_1 = simulate_agent(m, agent, epoch)
            df1 = pd.DataFrame(res_1[:-1,0:4], dtype='int',
                                columns=['Agent','Age','Job','PP'])
            df2 = pd.DataFrame(res_1[:-1,4:9], dtype='float',
                                columns=['log_h_cum','l','c','log_w', 'shock']) 
            df = pd.concat([df1,df2], axis=1)
            _a = min(_a, res_1[-2,8])
            _h = min(_h, np.exp(res_1[-2,4]))
            if epoch == 0 and agent == 1:
                _mode = 'w'
                _header = True
            else:
                _mode = 'a'
                _header = False   
            _filename = "data/results_1970.csv"                  
            df.to_csv(_filename, index=False, header =_header, mode= _mode)