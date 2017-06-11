import numpy as np
import random as rnd
import pandas as pd



from model import Model

from parameters import JOBS, PERIODS, AGENTS, \
                       AUGMA, NBA, AMIN, AMAX, AUGMH, NBH, HMIN, HMAX

from constants import ZSHOCKS, z_shock_range, pp_range, o_range

from helper import map_to_index, augm

from distributions import ZDISTRIBUTION, PPDDISTRIBUTION



# Moments of  model (All):
PSI = [0.41, 0.39, 0.27]                         # Labor supply elasticity
CHI = [0.36, 0.37, 0.44]                         # Disutility of labor supply



def simulate_agent(m, a, log=True):
    _dr_idx = NBA // 2
    _h = HMIN + (_dr_idx // NBA)*m.DELTAH
    _a = AMIN + (_dr_idx % NBA)*m.DELTAA
    res = np.array( [(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)]*PERIODS )
    if log:
        print(" Period| Job| PP| Shock|     h|     a|    l|     c|     w ")
        print(" {0:6d}|   -|  -|     -|{1:6.0f}|{2: 6.0f}|    -|     -|     -".
                format(0, _h, _a) )
    _chosen = -1
    _maxpp = len(PPDDISTRIBUTION[0])
    for period in range(PERIODS):            
        _temp_v = [0]*JOBS
        _temp_idx = [0]*JOBS
        _temp_state = [(0,0,0)]*JOBS
        for j in range(JOBS):
            _pp_prob = rnd.random()
            if period >= _maxpp:
                _per = _maxpp - 1
            else:
                _per = period
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
            _temp_idx[j] = m.dr[map_to_index(pp, j, 
                j != _chosen)].reshape(NBA*NBH, ZSHOCKS)[_dr_idx, _z_index]
            _temp_v[j] = m.v[map_to_index(pp, j, 
                j != _chosen)].reshape(NBA*NBH, ZSHOCKS)[_temp_idx[j], _z_index]
            _temp_state[j] = (pp, j == _chosen, z_shock_range[_z_index])
            
        _chosen = np.argmax(_temp_v)
        _l = m.labor(HMIN + (_temp_idx[_chosen] // NBA)*m.DELTAH, 
                    HMIN + (_dr_idx // NBA)*m.DELTAH) / AUGMH
        _w = m.wage( _chosen, 
                    _temp_state[_chosen][0], 
                    _l*AUGMH, 
                    HMIN + (_dr_idx // NBA)*m.DELTAH, 
                    _temp_state[_chosen][1],
                    _temp_state[_chosen][2] )
        _c = m.consumption(AMIN + (_temp_idx[_chosen] % NBA)*m.DELTAA,
                AMIN + (_dr_idx % NBA)*m.DELTAA, _w)
        _dr_idx = _temp_idx[_chosen]
        _h = HMIN + (_dr_idx // NBA)*m.DELTAH
        _a = AMIN + (_dr_idx % NBA)*m.DELTAA
        
        res[period] = (a, period + 1, _chosen, 
                    _temp_state[_chosen][0], 
                    _h, _l*AUGMH, _c, _w)
                    
        if log:
            print(" {0:6d}|{1: 4d}|{2: 3d}|{3: 6.2f}|".
                    format(a, period + 1,
                    _chosen, 
                    _temp_state[_chosen][0], 
                    _temp_state[_chosen][2]) +
                    "{0:6.0f}|{1: 6.0f}|{2: 5.3f}|{3: 6.0f}|{4: 6.0f}".
                        format(_h, _a, _l, _c, _w) )
    return res
        
def integrate(m):
    average = np.array( [(0.0, 0.0, 0.0, 0.0, 0.0)]*PERIODS )
    for agent in range(AGENTS):
        average += simulate_agent(m, log=False)
    average /= AGENTS
    print("     h|     a|    l|       c|       w ")
    for period in range(PERIODS):   
        (_h, _a, _l, _c, _w) = tuple(average[period])       
        print("{0:6.0f}|{1: 6.0f}|{2: 5.2f}|{3: 8.0f}|{4: 8.2f}".
                        format( _h, _a, _l, _c, _w ) )
    return average


m = Model(PSI[0], CHI[0], 0)
m.load_from_csv(1)
res_1 = simulate_agent(m, 0, log=False)
df1 = pd.DataFrame(res_1[:,0:4], dtype='int',columns=['Agent','Period','Job','PP'])
df2 = pd.DataFrame(res_1[:,4:], dtype='float',columns=['h_cum','h','c','w'])
df = pd.concat([df1,df2], axis=1)

for group in [0]:
    print(group+1)
    m = Model(PSI[group], CHI[group], group)
    m.load_from_csv(group + 1)
    for agent in range(1,AGENTS):
        res_1 = simulate_agent(m, agent + group*AGENTS, log=False)
        df1 = pd.DataFrame(res_1[:,0:4], dtype='int',columns=['Agent','Period','Job','PP'])
        df2 = pd.DataFrame(res_1[:,4:], dtype='float',columns=['h_cum','h','c','w'])
        df = df.append(pd.concat([df1,df2], axis=1))
            
df.to_csv("Group1.csv")