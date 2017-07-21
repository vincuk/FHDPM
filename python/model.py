import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import pandas as pd
from multiprocessing import Pool, cpu_count
from python.earnings import earnings_constants
from python.parameters import JOBS, CRITERION, MAXITERATIONS, \
                       AUGMA, NBA, AMIN, AMAX, AUGMH, NBH, HMIN, HMAX
from python.constants import BETA, R, IOTA, TAU, ZSHOCKS, pp_range, o_range
from python.rouwen import rouwen

class Model:
    '''
    Model
    '''

    def __init__(self, psi, chi, group,
                    a_min=AMIN, a_max=AMAX, nb_a=NBA, augm_a=AUGMA,
                    h_min=HMIN, h_max=HMAX, nb_h=NBH, augm_h=AUGMH):
        '''
        :param float psi: Labor supply elasticity
        :param float chi: Disutility of labor supply
        :param int group: Dimension of heterogeneity
        :param float a_min: Minimal value of assets
        :param float a_max: Maximal value of assets
        :param int nb_a: Number of assets points in the grid
        :param float augm_a: Borrowing limit 
        :param float h_min: Minimal value of hours worked
        :param float h_max: Maximal value of hours worked
        :param int nb_h: Number of hours worked points in the grid
        :param float augm_h: Maximum hours worked per year
        '''
        self.silent = True
        self.bond = [0.2, 0.2]
        self.PSI, self.CHI = (psi, chi)
        self.AUGMA = augm_a
        self.AUGMH = augm_h
        self.AMIN, self.AMAX, self.NBA = (a_min, a_max, nb_a)
        self.HMIN, self.HMAX, self.NBH = (h_min, h_max, nb_h)
        self.JOBS = JOBS
        self.CRITERION = CRITERION
        self.MAXITERATIONS = MAXITERATIONS
        self.BETA = BETA
        self.R =  R
        self.IOTA = IOTA
        self.TAU = TAU
        self.ZSHOCKS = ZSHOCKS
        self.z_shock_range = z_shock_range
        self.pp_range = pp_range
        self.o_range = o_range
        self.PIMATRIX, self.z_shock_range = rouwen(0.9, 0, 1/3, 3)
        self.WAGECONSTANT = earnings_constants[group]['WAGECONSTANT']
        self.ALPHA = earnings_constants[group]['ALPHA']
        self.ZETA = earnings_constants[group]['ZETA']
        self.GAMMA = earnings_constants[group]['GAMMA']
        self.XI = earnings_constants[group]['XI']
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
    
    def wage(self, j, pp, h, h_cum, o, z_shock):
        '''
        Earnings function
        
        :param int j: current job type
        :param int pp: 1 = receiving PP, 0 = not receiving PP
        :param float h: current hours worked
        :param float h_cum: cumulative hours worked
        :param bool o: True = changed job, False = kept job
        :param float z_shock: shock
        '''
        if h == 0:
            return 0
        else:
            _log_h = np.log(h)
        if o:
            _o = 1
        else:
            _o = 0
        return np.exp(
                    self.WAGECONSTANT + 
                    self.ALPHA*pp + 
                    self.ZETA[j] + 
                    self.GAMMA[0]*_log_h +
                    self.GAMMA[1]*np.log(h_cum) + 
                    self.GAMMA[2]*np.log(h_cum)*pp + 
                    self.XI*_o +
                    z_shock
                    )

    def consumption(self, a_prime, a, w):
        '''
        Consumption
        
        :param float a_prime: next period assets
        :param float a: current peruod assets
        :param float w: wage
        '''
        return (1 + self.R)*a - a_prime + (1 - self.TAU)*w
                    
    def labor(self, h_prime, h):
        '''
        Labor supply in hours
        
        :param float h_prime: next period cumulative hours worked
        :param float h: current peruod cumulative hours worked
        '''
        return h_prime - h    

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
        if h > self.HMAX - self.bond[0]*self.AUGMH:
            _h = self.bond[1]*self.AUGMH
        else:
            _h = h_prime - h
        return self.wage(j, pp, _h, h, o, self.z_shock_range[z_shock])

    def grid_consumption(self, j, pp, h_prime, h, a_prime, a, o, z_shock):
        '''
        Consumption on grid
    
        :param int j: current job type
        :param int pp: 1 = receiving PP, 0 = not receiving PP
        :param int h_prime: next period cumulative hours worked
        :param int h: current peruod cumulative hours worked
        :param float a_prime: next period assets
        :param float a: current peruod assets
        :param bool o: True = changed job, False = kept job
        :param int z_shock: z-shock index
        '''
        _w = self.grid_wage(j, pp, h_prime, h, o, z_shock)
        return self.consumption(a_prime, a, _w)

    def grid_labor(self, h_prime, h):
        '''
        Labor supply on grid
    
        :param int h_prime: next period cumulative hours worked
        :param int h: current peruod cumulative hours worked
        '''
        return self.labor(h_prime, h) / self.AUGMH
        
    def utility(self, c, l):
        '''
        Utility function on grid

        :param float c: consumption on grid
        :param float l: labor supply on grid
        '''
        _c = c
        return (_c**(1-self.IOTA))/(1-self.IOTA) - \
               self.CHI*(l**(1+self.PSI))/(1+self.PSI)
        
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
                                _l = self.bond[1]
                            else:
                                _l = self.grid_labor(h_end, h_start)
                            _idx_1 = self.map_from_grid(a_start, h_start)
                            _idx_2 = self.map_from_grid(a_end, h_end)
                            _temp_um[_idx_1, _idx_2, z_shock] = \
                                     self.utility(_c, _l)
        if not self.silent:
            print("Matrix: {0:2d}; Utility matrix calculated.".format(index))
        return _temp_um

    def iterate_model(self, index):
        _matrix_size = self._full_size*self.ZSHOCKS
        _temp_u0 = np.zeros((_matrix_size), dtype=float)
        _distance = 1
        _iteration = 0
        util = self.utility_matrix(index)
        while _distance > self.CRITERION and _iteration < self.MAXITERATIONS:
            _dr = np.nanargmax( util + 
                        self.BETA*np.tile( 
                                         self.v[index].dot(self.PIMATRIX), 
                                         (self._full_size, 1, 1)
                        ), axis=1 
                  ) 
            _Q = sp.lil_matrix((_matrix_size, _matrix_size), dtype=float)
            for z_shock in range(self.ZSHOCKS):
                _Q0 = sp.lil_matrix((self._full_size, self._full_size), 
                                    dtype=float)
                for i in range(self._full_size):
                    _Q0[i, _dr[i, z_shock]] = 1
                    _temp_u0[i + z_shock*self._full_size] = \
                        util[i, _dr[i, z_shock], z_shock]
                _Q[z_shock*self._full_size:(z_shock + 1)*self._full_size, :] = \
                        sp.kron(self.PIMATRIX[:, z_shock], _Q0)            
            _solution = spsolve( 
                               (sp.eye(_matrix_size) - self.BETA*_Q), _temp_u0 
                        )
            _temp_v = _solution.reshape((self.ZSHOCKS, self._full_size)).T
            _distance = np.max(abs(self.dr[index] - _dr))
            _iteration +=1
            self.v[index] = _temp_v
            self.dr[index] = _dr
        if not self.silent:
            print("Matrix: {0:2d}; Iteration: {1:2d}; Distance: {2:d}".
                format(index, _iteration, _distance))
        if _iteration == self.MAXITERATIONS:
            print("Max number of iterations reached!")
        return (_dr, _temp_v)

    def evaluate_model(self):
        '''
        Run multiprocessing computations
        '''
        CPU = cpu_count()
        pool = Pool(processes = CPU)
        res = [pool.apply_async(self.iterate_model, (i,)) \
                                                for i in range(self._size)]
        for i in range(self._size):
            (self.dr[i], self.v[i]) = res[i].get()
        
        pool.close()
        pool.terminate()
        pool.join()
        
    def map_to_index(self, pp, j, o):
        '''
        Return state index
    
        :param int pp: 1 = receiving PP, 0 = not receiving PP
        :param int j: current job type
        :param bool o: True = changed job, False = kept job
        '''
        if o:
            _o = 1
        else:
            _o = 0            
        return pp*( len(self.o_range)*self.JOBS ) + j*len(self.o_range) + _o
    
    def map_from_index(self, index):
        '''
        Return state from index
        
        :param int index: index
        '''
        if len(self.o_range) == 1:
            _o = self.o_range[0]
            if _o:
                _last = 1
            else:
                _last = 0 
            _j = (index - _last) % self.JOBS
        else:
            if index % len(self.o_range) == 0:
                _o = False
                _j = index % (len(self.o_range)*self.JOBS) // len(self.o_range)
            else:
                _o = True
                _j = (index % (len(self.o_range)*self.JOBS) - 1) // len(self.o_range)
        _pp = index // (len(self.o_range)*self.JOBS)
        return (_pp, _j, _o)
        
    def map_from_grid(self, a, h):
        a_idx = int(round((a - self.AMIN) / self.DELTAA))
        h_idx = int(round((h - self.HMIN) / self.DELTAH))
        return a_idx + h_idx*self.NBA
    
    def save_to_csv(self, fn):
        '''
        Save results to files
        '''
        _size = self._size * self._full_size
        _df = pd.DataFrame(np.array(self.dr).reshape(_size, self.ZSHOCKS),
                                columns=z_shock_range)
        _df.to_csv("data/dr-" + str(fn) + ".csv")
        _df = pd.DataFrame(np.array(self.v).reshape(_size, self.ZSHOCKS),
                                columns=z_shock_range)
        _df.to_csv("data/v-" + str(fn) + ".csv")
        
    def load_from_csv(self, fn):
        '''
        Load results from files
        '''
        _df = pd.read_csv("data/dr-" + str(fn) + ".csv", index_col=0)
        self.dr = _df.as_matrix().reshape(self._size, 
                            self._full_size, self.ZSHOCKS)
        _df = pd.read_csv("data/v-" + str(fn) + ".csv", index_col=0)
        self.v = _df.as_matrix().reshape(self._size, 
                            self._full_size, self.ZSHOCKS)