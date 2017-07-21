import numpy as np
import random as rnd
import pandas as pd

from python.model import Model
from python.parameters import CRITERION, MAXITERATIONS
from python.constants import BETA, R, IOTA, TAU, pp_range, o_range
from python.distributions import ZDISTRIBUTION, JOBDIST
from python.rouwen import rouwen


PERIODS = 27                               # Number of periods per one epoch
EPOCHS = 1                                 # Number of epochs  
AGENTS = 5000                              # Number of agents


class NewModel(Model):
    def __init__(self):
        self.silent = False
        self.bond = [0.16, 0.15, 0.01]
        self.CRITERION = CRITERION
        self.MAXITERATIONS = MAXITERATIONS
        self.JOBS = 9
        
        self.BETA = BETA
        self.R =  R
        self.IOTA = IOTA
        self.TAU = TAU  # 0.73 #
        self.ZSHOCKS = 3
        self.pp_range = pp_range
        self.o_range = o_range
        self.PIMATRIX, self.z_shock_range = rouwen(0.97, 0, 0.034, 3)   
        self.ZDISTRIBUTION = ZDISTRIBUTION
        self.AUGMA = 22000
        self.AUGMH = 5600
        self.AMIN, self.AMAX, self.NBA = (10, 150000, 50)
        self.HMIN, self.HMAX, self.NBH = (100, 5600, 55)
        self.PSI = 0.45
        self.CHI = [0.034, 0.028]
        self.WAGECONSTANT = 3.1
        self.ALPHA = 0.0066
        self.ZETA= [0, -0.31, -0.57, -0.28, -0.41, -0.55, -0.35, -0.63, -0.80]
        self.GAMMA = [0.77, 0.26, 0.062]
        self.XI = -0.09        
        self.updade_parameters()
        
    def utility(self, c, l, pp):
        '''
        Utility function on grid

        :param float c: consumption on grid
        :param float l: labor supply on grid
        '''
        _c = c
        return (_c**(1-self.IOTA))/(1-self.IOTA) - \
               self.CHI[pp]*(l**(1+self.PSI))/(1+self.PSI)
        
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
                            if h_start > self.HMAX - self.bond[0]*self.AUGMH:
                                _l = self.bond[1] + pp*self.bond[2]
                            else:
                                _l = self.grid_labor(h_end, h_start)
                            _idx_1 = self.map_from_grid(a_start, h_start)
                            _idx_2 = self.map_from_grid(a_end, h_end)
                            _temp_um[_idx_1, _idx_2, z_shock] = \
                                     self.utility(_c, _l, pp)
        if not self.silent:
            print("Matrix: {0:2d}; Utility matrix calculated.".format(index))
        return _temp_um

def simulate_agent(m, agent, epoch, init):
    _a, _h, pp, _chosen = init
    _dr_idx = m.map_from_grid(_a, _h)
    res = np.array( [(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]*PERIODS )
    for period in range(PERIODS):
        _temp_v = [0]*m.JOBS
        _temp_idx = [0]*m.JOBS
        _temp_shock = [0]*m.JOBS 
        _hired = False
        if _chosen == -1: # if the agent is unemployed 
            for j in range(m.JOBS):
                _employed_prob = rnd.random()
                if (_employed_prob < _tm[j, 4]):
                    _hired = True
                    _z_shock_prob = rnd.random()
                    _z_index = 0
                    _sum = m.ZDISTRIBUTION[_z_index]
                    while (_sum < _z_shock_prob):
                        _z_index += 1
                        _sum += m.ZDISTRIBUTION[_z_index]
                    _temp_idx[j] = m.dr[m.map_to_index(pp, j, True)].reshape(
                                m.NBA*m.NBH, m.ZSHOCKS)[_dr_idx, _z_index]
                    _temp_v[j] = m.v[m.map_to_index(pp, j, True)].reshape(
                             m.NBA*m.NBH, m.ZSHOCKS)[_temp_idx[j], _z_index]
                    _temp_shock[j] = m.z_shock_range[_z_index]
                else:
                    _temp_v[j] = -np.inf
            if _hired:
                _chosen = np.argmax(_temp_v)
                _dr_idx = _temp_idx[_chosen]
                _o, _z_shock = True, _temp_shock[_chosen]
                
        else: # if the agent is employed 
            _layoff_prob = rnd.random()
            if (_layoff_prob < _tm[_chosen, 2]):    
                _chosen = -1
            else:
                _hired = True
                _pp_prob = rnd.random() # updating PP status
                if pp == 1:
                    if (_pp_prob < _tm[_chosen, 0]):
                        pp = 0
                    else:
                        pp = 1
                else:
                    if (_pp_prob < _tm[_chosen, 1]):
                        pp = 1
                    else:
                        pp = 0
                if (_layoff_prob < _tm[_chosen, 2] + _tm[_chosen, 3]): # taking a new job            
                    for j in range(m.JOBS):
                        _z_shock_prob = rnd.random()
                        _z_index = 0
                        _sum = m.ZDISTRIBUTION[_z_index]
                        while (_sum < _z_shock_prob):
                            _z_index += 1
                            _sum += m.ZDISTRIBUTION[_z_index]
                        if (j != _chosen):
                            _temp_idx[j] = m.dr[m.map_to_index(pp, j, j != _chosen)].reshape(
                                m.NBA*m.NBH, m.ZSHOCKS)[_dr_idx, _z_index]
                            _temp_v[j] = m.v[m.map_to_index(pp, j, j != _chosen)].reshape(
                                m.NBA*m.NBH, m.ZSHOCKS)[_temp_idx[j], _z_index]
                            _temp_shock[j] = m.z_shock_range[_z_index]
                        else:
                            _temp_v[j] = -np.inf
                    _chosen = np.argmax(_temp_v)
                    _dr_idx = _temp_idx[_chosen]
                    _o, _z_shock = True, _temp_shock[_chosen]
                else: # staying at current job
                    _z_shock_prob = rnd.random()
                    _z_index = 0
                    _sum = m.ZDISTRIBUTION[_z_index]
                    while (_sum < _z_shock_prob):
                        _z_index += 1
                        _sum += m.ZDISTRIBUTION[_z_index]
                    _dr_idx = m.dr[m.map_to_index(pp, _chosen, False)].reshape(
                                m.NBA*m.NBH, m.ZSHOCKS)[_dr_idx, _z_index]
                    _o, _z_shock = False, m.z_shock_range[_z_index]
        
        if _hired:
            _l =  m.grid_labor(m.h_grid[_dr_idx // m.NBA], _h)
            _w = m.wage(_chosen, pp, _l*m.AUGMH, _h, _o, _z_shock)
            _c = m.consumption(m.a_grid[_dr_idx % m.NBA], _a, _w)
        else:
            _l = 0
            _w = 0
            _c = m.consumption(_a, _a, _w)
            _z_shock = np.nan
        if _w == 0:
            _lw = 0
        else:
            _lw = np.log(_w)        
        _h = m.h_grid[_dr_idx // m.NBA]
        _a = m.a_grid[_dr_idx % m.NBA]
        
        res[period] = (agent, epoch*(PERIODS-2) + period + 25, 
                    _chosen + 1, pp, np.log(_h), _l, _c, _lw, _z_shock, _a)
    return res
        
if __name__ == '__main__':
    _tm = pd.read_csv("data/transition_probs.csv", index_col=0).as_matrix()[:,2:]
    _h = 2500
    _a = 2000
    initials = [(0,0,0,0)]*AGENTS
    for agent in range(AGENTS):
        _pp_prob = rnd.random()
        if (_pp_prob < 0.2353642):
            _pp = 1
        else:
            _pp = 0
        _job_prob = rnd.random()
        _job = 0
        _sum = JOBDIST[_job]
        while (_sum < _job_prob):
            _job += 1
            _sum += JOBDIST[_job]
        initials[agent] = _a*rnd.randint(8, 12)/10, _h*rnd.randint(7, 14)/10, _pp, _job
    for epoch in range(EPOCHS):
        m = NewModel()
        m.NBH = 250
        m.HMIN = _h
        m.HMAX = m.HMIN + m.AUGMH*7.0
        m.NBA = 10
        m.AMIN = _a
        m.AMAX = m.AMIN + 70000
        print("Epoch: ", epoch)
        print("[", m.HMIN, m.HMAX, "] [", m.AMIN, m.AMAX, "]")
        m.updade_parameters()
        m.evaluate_model()
        _a = m.AMAX
        _h = m.HMAX
        for agent in range(1, AGENTS + 1):
            res_1 = simulate_agent(m, agent, epoch, initials[agent-1])
            df1 = pd.DataFrame(res_1[:-2,0:4], dtype='int',
                                columns=['Agent','Age','Job','PP'])
            df2 = pd.DataFrame(res_1[:-2,4:9], dtype='float',
                                columns=['log_h_cum','l','c','log_w', 'shock']) 
            df = pd.concat([df1,df2], axis=1)
            initials[agent-1] = res_1[-3,9], np.exp(res_1[-3,4]), res_1[-3,3], res_1[-3,2]
            _a = min(_a, initials[agent-1][0])
            _h = min(_h, initials[agent-1][1])
            if epoch == 0 and agent == 1:
                _mode = 'w'
                _header = True
            else:
                _mode = 'a'
                _header = False   
            if m.TAU > 0.7:
                _filename = "data/results_bigtau.csv"  
            else:
                _filename = "data/results.csv"                  
            df.to_csv(_filename, index=False, header =_header, mode= _mode)