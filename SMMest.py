import numpy as np
import random as rnd
import scipy.optimize as opt

from python.model import Model
from python.parameters import CRITERION, MAXITERATIONS
from python.constants import BETA, R, IOTA, TAU, pp_range, o_range, z_shock_range
from python.distributions import PPDDISTRIBUTION, ZDISTRIBUTION, PIMATRIX


PERIODS = 4                           # Number of periods for simulation
AGENTS = 5000                         # Number of agents

history = {}

class NewModel(Model):
    def __init__(self):
        self.bound = [0.2, 0.2, 0.1]
        self.AUGMA = 22000
        self.AUGMH = 4320
        
        # (self.AMIN, self.AMAX, self.NBA) = (0, 200000, 2*PERIODS)
        # (self.HMIN, self.HMAX, self.NBH) = (8900, 189000, 3*PERIODS)
        
        self.HMIN = 10000
        self.HMAX = self.HMIN + self.AUGMH*PERIODS 
        self.NBH = 5*PERIODS
        
        self.AMIN = 000
        self.AMAX = self.AMIN + 200000
        self.NBA = 3*PERIODS
        
        self.JOBS = 9
        self.CRITERION = CRITERION
        self.MAXITERATIONS = MAXITERATIONS
        self.BETA = BETA
        self.R =  R
        self.IOTA = IOTA
        self.TAU = TAU
        self.ZSHOCKS = 3
        self.z_shock_range = z_shock_range / 3.0
        self.pp_range = pp_range
        self.o_range = o_range # [False]
        self.PIMATRIX = PIMATRIX # np.array([[1]])
        
        self.PSI = 0.4
        self.CHI = 0.04
        self.WAGECONSTANT = 2.67
        self.ALPHA = 0.00456
        self.ZETA= [0, -0.31, -0.57, -0.28, -0.41, -0.55, -0.35, -0.63, -0.80]
        self.GAMMA = [0.7, 0.265, 0.027]
        self.XI = 0
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




def data_moments(dat):
    # output = np.array([10.030340, 8.260595, 10.325300, 8.469193])
    # output = np.array([10.2315, 9.4962, 10.6445, 9.7138])
    # output = np.array([ 9.214141, 9.393023, 
    #                     9.308854, 9.529869,
    #                     9.384467, 9.599511])

    output = np.array([10.20222, 9.214141, 10.46265, 9.393023, 
                        10.2162, 9.308854, 10.51371, 9.529869,
                        10.23466, 9.384467, 10.52668, 9.599511])
    return output
    
def sim_moments(params):
    key = str('{0:1.4f}-{1:1.4f}-{2:1.4f}-{3:1.4f}-{4:1.4f}').format(params.tolist()[0], 
                params.tolist()[1], params.tolist()[2], params.tolist()[3], 
                params.tolist()[4])
                
    # key = str('{0:1.4f}-{1:1.4f}-').format(params.tolist()[0], 
    #             params.tolist()[1])

    print(key)
    if key not in list(history.keys()):
        m = NewModel()

        # m.AMIN = params[0]*10000
        # m.AMAX = params[1]*100000
        
        m.WAGECONSTANT = params[0]*10.0
        m.ALPHA = params[1]/100.0
        m.GAMMA[0] = params[2]
        m.GAMMA[1] = params[3]
        m.GAMMA[2] = params[4]/10.0
        
        # m.PSI = params[0]
        # m.CHI = params[1]/10.0
        # m.bond = [0.2, params[1], params[2]]
        
        m.updade_parameters()
        m.evaluate_model()
        (res0, res1) = integrate(m)
        history[key] = np.array([res0[0,3], res0[0,2], res1[0,3], res1[0,2], 
                                res0[1,3], res0[1,2], res1[1,3], res1[1,2],
                                res0[2,3], res0[2,2], res1[2,3], res1[2,2]])
                                
        # history[key] = np.array([res0[0,2], res1[0,2], 
        #                          res0[1,2], res1[1,2],
        #                          res0[2,2], res1[2,2]])    
        
    return history[key]
    


def err_vec(data_vals, sim_params, simple):
    moms_data = data_moments(data_vals)
    moms_model = sim_moments(sim_params)
    print(moms_model)
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data
    
    return err_vec

def criterion(params, *args):
    xvals  = args
    err = err_vec(xvals, params, simple=True)
    crit_val = np.dot(err.T, err) 
    print("crit = ", crit_val)
    return crit_val

if __name__ == '__main__':
    # crit_test = criterion(np.array([3.36, 0.0, 0.758, 0.183, 0.026, 0.36]), 0)
    # print("crit_test = ", crit_test)
    # result: 0.0346923

    # crit_test = criterion(np.array([3.35, 0.003, 0.72, 0.13, 0.04, 0.36]), 0)
    # print("crit_test = ", crit_test)
    # result: 0.00565608
    
    # crit_test = criterion(np.array([3.35, 0.003, 0.72, 0.13, 0.04, 0.36]), 0)
    # print("crit_test = ", crit_test)
    # result: 0.0027860599
    
    # crit_test = criterion(np.array([3.35, 0.0040, 0.72, 0.13, 0.039, 0.359]), 0)
    # print("crit_test = ", crit_test)
    # result: 0.0122023226786

    # crit_test = criterion(np.array([3.3576, 0.004095, 0.735337, 0.13995, 0.0170, 0.35865]), 0)
    # print("crit_test = ", crit_test)
    # result: 0.31481 (True)
    
    # crit_test = criterion(np.array([0.33576, 0.4095, 0.735337, 0.13995, 0.170, 0.35865, 0.5]), 0)
    # print("crit_test = ", crit_test)
    # result: 0.555275 (True)
    
    # crit_test = criterion(np.array([0.359, 0.41]), 0)
    # print("crit_test = ", crit_test)
    # result: 0.21135 (True)
    
    # crit_test = criterion(np.array([0.27, 0.456, 0.7, 0.2688, 0.266]), 0)
    # print("crit_test = ", crit_test)
    # result: 0.18597976 (True)
    
    # crit_test = criterion(np.array([1, 5]), 0)
    # print("crit_test = ", crit_test)
    # result: 0.202696 (True)
    
    # crit_test = criterion(np.array([0.267, 0.456, 0.7, 0.265, 0.27]), 0)
    # print("crit_test = ", crit_test)
    # result: 0.18597976 (True)
    
    params_init = np.array([0.267, 0.456, 0.7, 0.265, 0.27]) 
    smm_args = (0)

    results = opt.minimize(criterion, params_init, args=(smm_args),
                          method='BFGS', 
                          options={'disp': True, 'maxiter' :10, 
                          'eps': 1e-3, 'gtol': 5e-3})

    print('res: ', results.x)