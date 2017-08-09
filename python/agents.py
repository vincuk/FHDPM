import numpy as np
import random as rnd

from python.parameters import PERIODS, AGENTS
from python.distributions import JOBDIST

def set_initial(a, h, random = False):
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
    if random:
        initials = a*rnd.randint(8, 12)/10, \
                   h*rnd.randint(7, 14)/10, \
                   _pp, _job
    else:
        initials = a, h, _pp, _job
    return initials
    
def simulate_agent(m, agent, initials):
    '''
    :param Model m: model object
    :param int agent: index of agent
    :param  (float, float, int, int) init: initial parameters = 
                                    (asset, hours worked, pp, job#)

    OUTPUT:
    (agent, age, job#, pp, log_h_cum, l, c, log_w, z_shock, a)
    '''
    _a, _h, pp, _chosen = initials
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
                if (_employed_prob < m.TM[j, 4]):
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
            if (_layoff_prob < m.TM[_chosen, 2]):    
                _chosen = -1
            else:
                _hired = True
                _pp_prob = rnd.random() # updating PP status
                if pp == 1:
                    if (_pp_prob < m.TM[_chosen, 0]):
                        pp = 0
                    else:
                        pp = 1
                else:
                    if (_pp_prob < m.TM[_chosen, 1]):
                        pp = 1
                    else:
                        pp = 0
                if (_layoff_prob < m.TM[_chosen, 2] + m.TM[_chosen, 3]): # taking a new job            
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
        
        res[period] = (agent, period + 25, _chosen + 1, pp, 
                        np.log(_h), _l, _c, _lw, _z_shock, _a)
    return res
    
def integrate_agents(m, a, h, random = False):
    average_pp_0 = np.array( [(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]*PERIODS )
    average_pp_1 = np.array( [(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]*PERIODS )
    index_pp_0 = np.array( [0]*(PERIODS) )
    index_pp_1 = np.array( [0]*(PERIODS) )

    for agent in range(AGENTS):
        initials = set_initial(a, h, random)
        _temp_agent = simulate_agent(m, 0, initials)
        for period in range(PERIODS):
            if _temp_agent[period][2] == 0:
                continue
            if _temp_agent[period][3] == 0:
                average_pp_0[period] += _temp_agent[period]
                index_pp_0[period] += 1
            else:
                average_pp_1[period] += _temp_agent[period]
                index_pp_1[period] += 1

    for period in range(PERIODS):
        average_pp_0[period] /= index_pp_0[period]
        average_pp_1[period] /= index_pp_1[period]
    return (average_pp_0, average_pp_1)