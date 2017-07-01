import numpy as np
import random as rnd
import pandas as pd

from python.model import Model
from python.parameters import CRITERION, MAXITERATIONS
from python.constants import BETA, R, IOTA, TAU, pp_range, o_range
from python.distributions import PPDDISTRIBUTION, ZDISTRIBUTION
from python.rouwen import rouwen


PERIODS = 6                               # Number of periods per one epoch
EPOCHS = 6                                # Number of epochs  
AGENTS = 7000                             # Number of agents


class NewModel(Model):
    def __init__(self):
        self.silent = False
        self.bond = [0.2, 0.2]
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
        self.PIMATRIX, self.z_shock_range = rouwen(0.9, 0, 1/3, 3)        
        self.AUGMA = 22000
        self.AUGMH = 4320
        self.AMIN, self.AMAX, self.NBA = (25000, 525000, 50)
        self.HMIN, self.HMAX, self.NBH = (8900, 189000, 50)
        self.PSI = 0.4
        self.CHI = 0.04
        self.WAGECONSTANT = 2.24
        self.ALPHA = 0.0064
        self.ZETA= [0, -0.31, -0.57, -0.28, -0.41, -0.55, -0.35, -0.63, -0.80]
        self.GAMMA = [0.71, 0.25, 0.089]
        self.XI = -0.05
        self.updade_parameters()

def simulate_agent(m, agent, epoch):
    _dr_idx = rnd.randint(0, 1)
    _max_period_pp = len(PPDDISTRIBUTION[0])
    _h = m.h_grid[_dr_idx // m.NBA]
    _a = m.a_grid[_dr_idx % m.NBA]
    res = np.array( [(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]*PERIODS )
    _chosen = -1
    for period in range(PERIODS):    
        if period >= _max_period_pp:
            _per = _max_period_pp - 1
        else:
            _per = period
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
            _temp_idx[j] = m.dr[m.map_to_index(pp, j, j == _chosen)].reshape(
                                m.NBA*m.NBH, m.ZSHOCKS)[_dr_idx, _z_index]
            _temp_v[j] = m.v[m.map_to_index(pp, j, j == _chosen)].reshape(
                             m.NBA*m.NBH, m.ZSHOCKS)[_temp_idx[j], _z_index]
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
    _h = 2300
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
            if m.TAU > 0.7:
                _filename = "data/results_bigtau.csv"  
            else:
                _filename = "data/results.csv"                  
            df.to_csv(_filename, index=False, header =_header, mode= _mode)