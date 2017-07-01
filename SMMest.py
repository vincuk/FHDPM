import numpy as np
import random as rnd
import scipy.optimize as opt
import pandas as pd

from python.model import Model
from python.parameters import CRITERION, MAXITERATIONS
from python.constants import BETA, R, IOTA, TAU, pp_range, o_range
from python.distributions import PPDDISTRIBUTION, ZDISTRIBUTION
from python.rouwen import rouwen

PERIODS = 6                         # Number of periods per one epoch
AGENTS = 25000                        # Number of agents

class NewModel(Model):
    def __init__(self):
        self.silent = True
        self.bond = [0.2, 0.2]
        self.CRITERION = CRITERION
        self.MAXITERATIONS = MAXITERATIONS
        self.JOBS = 9

        self.BETA = BETA
        self.R =  R
        self.IOTA = IOTA
        self.TAU = TAU
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

def simulate_agent(m):
    _dr_idx = rnd.randint(0, 1)
    _maxpp = len(PPDDISTRIBUTION[0])
    _h = m.h_grid[_dr_idx // m.NBA]
    res = np.array( [(0, 0, 0.0, 0.0)]*PERIODS )
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
        _dr_idx = _temp_idx[_chosen]
        _h = m.h_grid[_dr_idx // m.NBA]
        res[period] = (period + 1, pp, np.log(_h), np.log(_w))
    return res

def integrate(m):
    average_pp_0 = np.array( [(0, 0, 0.0, 0.0)]*(PERIODS) )
    average_pp_1 = np.array( [(0, 0, 0.0, 0.0)]*(PERIODS) )
    index_pp_0 = np.array( [0]*(PERIODS) )
    index_pp_1 = np.array( [0]*(PERIODS) )
    for agent in range(AGENTS):
        _temp_agent = simulate_agent(m)
        for period in range(PERIODS):
            if _temp_agent[period][1] == 0:
                average_pp_0[period] += _temp_agent[period]
                index_pp_0[period] += 1
            else:
                average_pp_1[period] += _temp_agent[period]
                index_pp_1[period] += 1
    for period in range(PERIODS):
        average_pp_0[period] /= index_pp_0[period]
        average_pp_1[period] /= index_pp_1[period]
    return (average_pp_0, average_pp_1)

def data_moments(periods):
    min_per, max_per = periods
    _df = pd.read_csv("data/full_set_moments.csv", index_col=0)
    output = np.concatenate(
                (_df[_df.pp == 0]['llabinc'].as_matrix()[min_per:max_per],
                _df[_df.pp == 1]['llabinc'].as_matrix()[min_per:max_per],
                _df[_df.pp == 0]['lcumul_hours'].as_matrix()[min_per:max_per],
                _df[_df.pp == 1]['lcumul_hours'].as_matrix()[min_per:max_per])
             )                 
    return output

history = {}

def sim_moments(params, *args):
    key = str('{0:1.4f} {1:1.4f} {2:1.4f} {3:1.4f} {4:1.4f} {5:1.4f} {6:1.4f} {7:1.4f}').format(
                params.tolist()[0], params.tolist()[1], params.tolist()[2], 
                params.tolist()[3], params.tolist()[4], params.tolist()[5],
                params.tolist()[6], params.tolist()[7]
                )          
    print("params: ", key)
    if key not in list(history.keys()):
        min_per, max_per = args[0]
        epochs = 1 + max_per // (PERIODS - 1)
        _h = 2300
        _a = 22000
        _llabinc0 = np.array([])
        _lcumul_hours0 = np.array([])
        _llabinc1 = np.array([])
        _lcumul_hours1 = np.array([])
        for epoch in range(epochs):
            m = NewModel()
            m.NBH = 3*PERIODS
            m.HMIN = _h
            m.HMAX = m.HMIN + m.AUGMH*PERIODS
            m.NBA = 2*PERIODS
            m.AMIN = _a
            m.AMAX = m.AMIN + 300000
            m.WAGECONSTANT = params[0]*10.0
            m.ALPHA = params[1]/100.0
            m.GAMMA[0] = params[2]
            m.GAMMA[1] = params[3]
            m.GAMMA[2] = params[4]/10.0
            m.XI = params[5]/10.0
            m.PSI = params[6]
            m.CHI = params[7]/10.0
            m.updade_parameters()
            m.evaluate_model()
            (res0, res1) = integrate(m)
            _llabinc0 = np.append(_llabinc0, res0[:-1, 3])
            _llabinc1 = np.append(_llabinc1, res1[:-1, 3])
            _lcumul_hours0 = np.append(_lcumul_hours0, res0[:-1, 2])
            _lcumul_hours1 = np.append(_lcumul_hours1, res1[:-1, 2])   
        history[key] = np.concatenate(
                        (_llabinc0[min_per:max_per], 
                        _llabinc1[min_per:max_per], 
                        _lcumul_hours0[min_per:max_per], 
                        _lcumul_hours1[min_per:max_per])
                        )            
    return history[key]
    
def err_vec(periods, sim_params, simple):
    moms_data = data_moments(periods)
    moms_model = sim_moments(sim_params, periods)
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data
    return err_vec

def criterion(params, *args):
    periods = args
    err = err_vec(periods, params, simple=False)
    crit_val = np.dot(err.T, err) 
    print("crit = ", crit_val)
    return crit_val

if __name__ == '__main__':    
    params_init = np.array([0.224, 0.64, 0.71, 0.25, 0.89, -0.5, 0.4, 0.4]) 
    periods = (7, 27)
    results = opt.minimize(criterion, params_init, args=(periods),
                          method='BFGS', 
                          options={'disp': True, 'maxiter' :10, 
                          'eps': 1e-3, 'gtol': 1e-1})
    print('WAGECONSTANT = {0:1.2f}'.format(results.x[0]*10.0))
    print('ALPHA = {0:1.4f}'.format(results.x[1]/100.0))
    print('GAMMA[0] = {0:1.2f}'.format(results.x[2]))
    print('GAMMA[1] = {0:1.2f}'.format(results.x[3]))
    print('GAMMA[2] = {0:1.3f}'.format(results.x[4]/10.0))
    print('XI = {0:1.3f}'.format(results.x[5]/10.0))
    print('PSI = {0:1.2f}'.format(results.x[6]))
    print('CHI = {0:1.3f}'.format(results.x[7]/10.0))