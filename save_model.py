import numpy as np
import random as rnd
import pandas as pd
from python.model import Model
from python.parameters import PERIODS, AGENTS
from python.distributions import ZDISTRIBUTION, PPDDISTRIBUTION

# Moments of  model (All):
PSI = [0.41, 0.39, 0.27]                         # Labor supply elasticity
CHI = [0.36, 0.37, 0.44]                         # Disutility of labor supply


def simulate_agent(m, agent, log=True):
    _dr_idx = rnd.randint(0, 5)*m.NBA + rnd.randint(m.NBA // 4, m.NBA // 2)
    _maxpp = len(PPDDISTRIBUTION[0])
    _h = m.HMIN + (_dr_idx // m.NBA)*m.DELTAH
    _a = m.AMIN + (_dr_idx % m.NBA)*m.DELTAA
    res = np.array( [(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)]*PERIODS )
    if log:
        print(" Period| Job| PP| Shock|     h|      a|    l|     c|     w ")
        print(" {0:6d}|   -|  -|     -|{1:6.0f}|{2: 6.0f}|    -|     -|     -".
                format(0, _h, _a) )
    _chosen = -1
    for period in range(PERIODS):    
        if period >= _maxpp:
            _per = _maxpp - 1
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
            _temp_idx[j] = m.dr[m.map_to_index(pp, j, 
                j == _chosen)].reshape(m.NBA*m.NBH, m.ZSHOCKS)[_dr_idx, _z_index]
            _temp_v[j] = m.v[m.map_to_index(pp, j, 
                j == _chosen)].reshape(m.NBA*m.NBH, m.ZSHOCKS)[_temp_idx[j], _z_index]
            _temp_state[j] = (pp, j == _chosen, m.z_shock_range[_z_index])
        _chosen = np.argmax(_temp_v)
        _l = m.labor(m.HMIN + (_temp_idx[_chosen] // m.NBA)*m.DELTAH, 
                    m.HMIN + (_dr_idx // m.NBA)*m.DELTAH) / m.AUGMH

        _w = m.wage( _chosen, 
                    _temp_state[_chosen][0], 
                    _l*m.AUGMH, 
                    m.HMIN + (_dr_idx // m.NBA)*m.DELTAH, 
                    _temp_state[_chosen][1],
                    _temp_state[_chosen][2] )
        
        _c = m.consumption(m.AMIN + (_temp_idx[_chosen] % m.NBA)*m.DELTAA,
                m.AMIN + (_dr_idx % m.NBA)*m.DELTAA, _w)
        _dr_idx = _temp_idx[_chosen]
        _h = m.HMIN + (_dr_idx // m.NBA)*m.DELTAH
        _a = m.AMIN + (_dr_idx % m.NBA)*m.DELTAA
        
        res[period] = (agent, period + 1, _chosen, 
                    _temp_state[_chosen][0], 
                    _h, _l*m.AUGMH, _c, _w)
        if log:
            print(" {0:6d}|{1: 4d}|{2: 3d}|{3: 6.2f}|".
                    format(period + 1,
                    _chosen, 
                    _temp_state[_chosen][0], 
                    _temp_state[_chosen][2]) +
                    "{0:6.0f}|{1: 6.0f}|{2: 5.4f}|{3: 6.0f}|{4: 6.0f}".
                        format(_h, _a, _l, _c, _w) )
    return res
        
if __name__ == '__main__':
    for group in [0, 1, 2]:
        print("Group: " + str(group+1))
        m = Model(PSI[group], CHI[group], group)
        m.load_from_csv(group + 1)
        for agent in range(1, AGENTS + 1):
            res_1 = simulate_agent(m, agent + group*AGENTS, log=False)
            df1 = pd.DataFrame(res_1[:,0:4], dtype='int',columns=['Agent','Period','Job','PP'])
            df2 = pd.DataFrame(res_1[:,4:], dtype='float',columns=['h_cum','h','c','w']) 
            df = pd.concat([df1,df2], axis=1)
            if agent == 1:
                _mode = 'w'
                _header = True
            else:
                _mode = 'a'
                _header = False        
            df.to_csv("data/Group" + str(group+1) + ".csv", 
                        index=False, header =_header, mode= _mode)