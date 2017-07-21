import numpy as np
import random as rnd
import scipy.optimize as opt
import pandas as pd

from python.model import Model
from python.parameters import CRITERION, MAXITERATIONS
from python.constants import BETA, R, IOTA, TAU, pp_range, o_range
from python.distributions import ZDISTRIBUTION, JOBDIST
from python.rouwen import rouwen

PERIODS = 29                               # Number of periods per one epoch
AGENTS = 5000                             # Number of agents

def custmin(fun, x0, args=(), maxfev=None, stepsize=0.01, 
            maxiter=20, callback=None, **options):
    bestx = x0
    besty = fun(bestx, *args)
    funcalls = 1
    niter = 0
    improved = True
    stop = False
    while improved and not stop and niter < maxiter:
        improved = False
        niter += 1
        for dim in range(np.size(x0)):
            for s in [bestx[dim] - stepsize, bestx[dim] + stepsize]:
                testx = np.copy(bestx)
                testx[dim] = s
                testy = fun(testx, *args)
                funcalls += 1
                if testy < besty:
                    besty = testy
                    bestx = testx
                    improved = True
            if callback is not None:
                callback(bestx)
            if maxfev is not None and funcalls >= maxfev:
                stop = True
                break
    return opt.OptimizeResult(fun=besty, x=bestx, nit=niter, nfev=funcalls, success=(niter > 1))


class NewModel(Model):
    def __init__(self):
        self.silent = True
        self.bond = [0.16, 0.15, 0.01]
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
        self.PIMATRIX, self.z_shock_range = rouwen(0.97, 0, 0.034, 3)   
        self.ZDISTRIBUTION = ZDISTRIBUTION     
        self.AUGMA = 22000
        self.AUGMH = 5600
        self.AMIN, self.AMAX, self.NBA = (30000, 180000, 10)
        self.HMIN, self.HMAX, self.NBH = (100, 8500, 300)
        self.PSI = 0.4
        self.CHI = [0.035, 0.030]
        self.WAGECONSTANT = 2.6
        self.ALPHA = 0.0065
        self.ZETA= [0, -0.31, -0.57, -0.28, -0.41, -0.55, -0.35, -0.63, -0.80]
        self.GAMMA = [0.71, 0.25, 0.059]
        self.XI = -0.089        
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

def simulate_agent(m, init):
    _a, _h, pp, _chosen = init
    _dr_idx = m.map_from_grid(_a, _h)
    res = np.array( [(0, 0, 0.0, 0.0, 0.0, 0.0)]*PERIODS )
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
        
        res[period] = (period + 1, pp, np.log(_h), _lw, _l, _a)
    return res

def integrate(m):
    average_pp_0 = np.array( [(0, 0, 0.0, 0.0, 0.0, 0.0)]*(PERIODS) )
    average_pp_1 = np.array( [(0, 0, 0.0, 0.0, 0.0, 0.0)]*(PERIODS) )
    index_pp_0 = np.array( [0]*(PERIODS) )
    index_pp_1 = np.array( [0]*(PERIODS) )
    _a = m.AMAX
    _h = m.HMAX
    for agent in range(AGENTS):
        _temp_agent = simulate_agent(m, initials[agent])
        for period in range(PERIODS):
            if _temp_agent[period][1] == 0:
                average_pp_0[period] += _temp_agent[period]
                index_pp_0[period] += 1
            else:
                average_pp_1[period] += _temp_agent[period]
                index_pp_1[period] += 1
        initials[agent] = _temp_agent[-2,5], np.exp(_temp_agent[-2,2])
        _a = min(_a, initials[agent][0])
        _h = min(_h, initials[agent][1])
    for period in range(PERIODS):
        average_pp_0[period] /= index_pp_0[period]
        average_pp_1[period] /= index_pp_1[period]
    return (average_pp_0, average_pp_1, (_a, _h))

def data_moments(periods):
    min_per, max_per = periods
    _df = pd.read_csv("data/full_set_moments.csv", index_col=0)
    output = np.concatenate(
                (_df[_df.pp == 0]['llabinc'].as_matrix()[min_per:max_per],
                _df[_df.pp == 1]['llabinc'].as_matrix()[min_per:max_per],
                _df[_df.pp == 0]['lcumul_hours'].as_matrix()[min_per:max_per],
                _df[_df.pp == 1]['lcumul_hours'].as_matrix()[min_per:max_per],
                _df[_df.pp == 0]['share_hours'].as_matrix()[min_per:max_per]*10,
                _df[_df.pp == 1]['share_hours'].as_matrix()[min_per:max_per]*10)
             )                 
    return output

def sim_moments(params, *args):
    global initials 
    key = str('{0:1.2f} {1:1.2f} {2:1.2f} {3:1.2f} {4:1.2f} {5:1.2f} {6:1.2f} {7:1.2f} {8:1.2f}').format(
                params.tolist()[0], params.tolist()[1], params.tolist()[2], 
                params.tolist()[3], params.tolist()[4], params.tolist()[5],
                params.tolist()[6], params.tolist()[7], params.tolist()[8]
                )   
    print("params: ", key)
    _f.write("params: " + key + "\n")
    if key not in list(history.keys()):
        min_per, max_per = args[0]
        epochs = 1
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
        _llabinc0 = np.array([])
        _lcumul_hours0 = np.array([])
        _share_hours0 = np.array([])
        _llabinc1 = np.array([])
        _lcumul_hours1 = np.array([])
        _share_hours1 = np.array([])
        for epoch in range(epochs):
            m = NewModel()
            m.NBH = 250
            m.HMIN = _h
            m.HMAX = m.HMIN + m.AUGMH*7.0
            m.NBA = 10
            m.AMIN = _a
            m.AMAX = m.AMIN + 70000
            m.WAGECONSTANT = params[0]*10.0
            m.ALPHA = params[1]/100.0
            m.GAMMA[0] = params[2]
            m.GAMMA[1] = params[3]
            m.GAMMA[2] = params[4]/10.0
            m.XI = params[5]/10.0
            m.PSI = params[6]
            m.CHI[0] = params[7]/10.0
            m.CHI[1] = params[8]/10.0
            m.updade_parameters()
            m.evaluate_model()
            (res0, res1, (_a, _h)) = integrate(m)
            _llabinc0 = np.append(_llabinc0, res0[:-1, 3])
            _llabinc1 = np.append(_llabinc1, res1[:-1, 3])
            _lcumul_hours0 = np.append(_lcumul_hours0, res0[:-1, 2])
            _lcumul_hours1 = np.append(_lcumul_hours1, res1[:-1, 2])   
            _share_hours0 = np.append(_share_hours0, res0[:-1, 4])
            _share_hours1 = np.append(_share_hours1, res1[:-1, 4])  
        history[key] = np.concatenate(
                        (_llabinc0[min_per:max_per], 
                        _llabinc1[min_per:max_per], 
                        _lcumul_hours0[min_per:max_per], 
                        _lcumul_hours1[min_per:max_per],
                        _share_hours0[min_per:max_per]*10, 
                        _share_hours1[min_per:max_per]*10)
                        )        
        del m    
    return history[key]
    
def err_vec(periods, sim_params, simple):
    moms_data = data_moments(periods)
    moms_model = sim_moments(sim_params, periods)
    _f.write("moms_data:\n" + str(moms_data) + "\n")
    _f.write("moms_model:\n" + str(moms_model) + "\n")
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
    _f.write("crit = " + str(crit_val) + "\n")

    return crit_val

history = {}

if __name__ == '__main__':    
    _f = open("data/SMM_log.txt", 'w')
    _tm = pd.read_csv("data/transition_probs.csv", index_col=0).as_matrix()[:,2:]

    params_init = np.array([0.31, 0.67, 0.74, 0.26, 0.62, -0.89, 0.46, 0.34, 0.28]) 
    periods = (3, 27)
 
    # criterion(params_init, periods)

    results = opt.minimize(criterion, params_init, args=(periods),
                          method='BFGS', 
                          options={'disp': True, 'maxiter' :10, 
                          'eps': 1e-2, 'gtol': 3e-2})
    # results = opt.minimize(criterion, params_init, args=(periods),
    #                       method=custmin, options=dict(stepsize=0.01))
    _f.write('WAGECONSTANT = {0:1.2f}'.format(results.x[0]*10.0) + "\n")
    _f.write('ALPHA = {0:1.4f}'.format(results.x[1]/100.0) + "\n")
    _f.write('GAMMA[0] = {0:1.2f}'.format(results.x[2]) + "\n")
    _f.write('GAMMA[1] = {0:1.2f}'.format(results.x[3]) + "\n")
    _f.write('GAMMA[2] = {0:1.3f}'.format(results.x[4]/10.0) + "\n")
    _f.write('XI = {0:1.3f}'.format(results.x[5]/10.0) + "\n")
    _f.write('PSI = {0:1.2f}'.format(results.x[6]) + "\n")
    _f.write('CHI 1 = {0:1.3f}'.format(results.x[7]/10.0) + "\n")
    _f.write('CHI 2 = {0:1.3f}'.format(results.x[8]/10.0) + "\n")
    _f.close()