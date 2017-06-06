import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import pandas as pd
import matplotlib.pyplot as plt

from helper import map_to_index

from earnings import earnings_constants

from parameters import JOBS, CRITERION, MAXITERATIONS, PERIODS, AGENTS, \
                       AUGMA, NBA, AMIN, AMAX, AUGMH, NBH, HMIN, HMAX
from constants import BETA, R, IOTA, TAU, ZSHOCKS, z_shock_range, pp_range, \
                      PPDDISTRIBUTION, PIMATRIX, ZDISTRIBUTION, o_range

class Model:
    '''
    Model
    '''

    def __init__(self, psi, chi, group):
        '''
        :param float psi: Labor supply elasticity
        :param float chi: Disutility of labor supply
        :param int group: Dimension of heterogeneity
        '''
        (self.PSI, self.CHI) = (psi, chi)
        
        self.DELTAA = (AMAX - AMIN) / NBA      # Increment of assets
        self.DELTAH = (HMAX - HMIN) / NBH      # Increment of hours worked
        
        self.WAGECONSTANT = earnings_constants[group]['WAGECONSTANT']
        self.ALPHA = earnings_constants[group]['ALPHA']
        self.ZETA = earnings_constants[group]['ZETA']
        self.GAMMA = earnings_constants[group]['GAMMA']
        self.XI = earnings_constants[group]['XI']
        
        self.a_grid = np.linspace(AMIN, AMAX, NBA)
        self.h_grid = np.linspace(HMIN, HMAX, NBH)
        self.util = []
        dr = np.zeros( (NBA*NBH, ZSHOCKS), dtype=int )
        v = np.zeros( (NBA*NBH, ZSHOCKS), dtype=float )
        self.v = [v] * ( len(pp_range)*len(o_range)*JOBS )
        self.dr = [dr] * ( len(pp_range)*len(o_range)*JOBS )    
    
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
        if o:
            _o = 1
        else:
            _o = 0
        if h == 0:
            _log_h = -np.inf
        else:
            _log_h = np.log(h)
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
        return (1 + R)*a - a_prime + (1 - TAU)*w
                    
    def labor(self, h_prime, h):
        '''
        Labor supply in hours
        
        :param float h_prime: next period cumulative hours worked
        :param float h: current peruod cumulative hours worked
        '''
        return (h_prime - h)    

    def grid_wage(self, j, pp, h_prime, h, o, z_shock):
        '''
        Earnings function on grid
        
        :param int j: current job type
        :param int pp: 1 = receiving PP, 0 = not receiving PP
        :param int h_prime: index of the next period cumulative hours worked
        :param int h: index of the current peruod cumulative hours worked
        :param bool o: True = changed job, False = kept job
        :param int z_shock: z-shock index
        '''
        if 2*h - h_prime < 0:
            _h_old = HMIN
        else:
            _h_old = HMIN + self.DELTAH*(2*h - h_prime)
        return self.wage( j, pp, self.DELTAH*(h_prime - h), _h_old, o, z_shock_range[z_shock] )
    
    def grid_consumption(self, j, pp, h_prime, h, a_prime, a, o, z_shock):
        '''
        Consumption on grid
    
        :param int j: current job type
        :param int pp: 1 = receiving PP, 0 = not receiving PP
        :param int h_prime: index of the next period cumulative hours worked
        :param int h: index of the current peruod cumulative hours worked
        :param float a_prime: index of the next period assets
        :param float a: index of the current peruod assets
        :param bool o: True = changed job, False = kept job
        :param int z_shock: z-shock index
        '''
        _w = self.grid_wage(j, pp, h_prime, h, o, z_shock)
        return self.consumption( AMIN + self.DELTAA*a_prime, 
                            AMIN + self.DELTAA*a, 
                            _w )
                
    def grid_labor(self, h_prime, h):
        '''
        Labor supply on grid
    
        :param int h_prime: index of the next period cumulative hours worked
        :param int h: index of the current peruod cumulative hours worked
        '''
        return self.DELTAH * self.labor(h_prime, h) / AUGMH
        
    def utility(self, c, l):
        '''
        Utility function on grid

        :param float c: consumption on grid
        :param float l: labor supply on grid
        '''
        return self.CHI*(c**(1-IOTA))/(1-IOTA) - \
               (1 - self.CHI)*(l**(1+self.PSI))/(1+self.PSI)
        
    def evaluate_model(self):
        if len(self.util) == 0:
            self.update_utility_matrix()
        self.iterate_model()
        
    def update_utility_matrix(self):
        print("Updating utility matrix", end='')
        for pp in pp_range:
            for j in range(JOBS):
                for o in o_range:
                    print(".", end='')
                    _temp_um = np.zeros((NBA*NBH, NBA*NBH, ZSHOCKS), dtype=float)
                    _temp_um.fill(np.nan) 
                
                    for z_shock in range(ZSHOCKS): 
                        for h_start in range(NBH):
                            for a_start in range(NBA):
                                for h_end in range(h_start, NBH):
                                    if h_end - h_start > AUGMH/self.DELTAH:
                                        continue
                                    for a_end in range(a_start, NBA):
                                        if a_end - a_start > AUGMA/self.DELTAA:
                                            continue
                                        _c = self.grid_consumption(j, pp, 
                                                            h_end, h_start, 
                                                            a_end, a_start,
                                                            o, z_shock);
                                        if _c < 0:
                                            continue
                                        _l = self.grid_labor(h_end, h_start)
                                        _temp_um[a_start + h_start*NBA, 
                                                    a_end + h_end*NBA, 
                                                    z_shock] = \
                                            self.utility(_c, _l)
                    self.util.append(_temp_um)
        print(" Done")

    def iterate_model(self):
        _temp_u0 = np.zeros( (NBA*NBH*ZSHOCKS), dtype=float )
        _distance = 1
        _iteration = 0
        while _distance > CRITERION and _iteration < MAXITERATIONS:
            _dr = []
            _temp_v = []

            for pp in pp_range:
                for j in range(JOBS):
                    for o in o_range:
                        _dr.append( np.nanargmax(
                                    self.util[map_to_index(pp, j, o)] + 
                                    BETA*np.tile( self.v[map_to_index(pp, j, o)]
                                    .dot(PIMATRIX), 
                                    (NBA*NBH, 1, 1)), axis=1 ) )
                        _Q = sp.lil_matrix( 
                                    (NBA*NBH*ZSHOCKS, NBA*NBH*ZSHOCKS), 
                                    dtype=float )

                        for z_shock in range(ZSHOCKS):
                            _Q0 = sp.lil_matrix( (NBA*NBH, NBA*NBH), 
                                                    dtype=float )
                            for i in range(NBA*NBH):
                                _Q0[i, _dr[map_to_index(pp, j, o)][i, 
                                                                z_shock]] = 1
                                _temp_u0[i + z_shock*NBA*NBH] = \
                                    self.util[map_to_index(pp, j, o)][i, 
                                        _dr[map_to_index(pp, j, o)][i, z_shock], 
                                        z_shock]
                            _Q[z_shock*NBA*NBH : (z_shock + 1)*NBA*NBH, :] = \
                                    sp.kron( PIMATRIX[:, z_shock], _Q0 )            

                        _temp_u0 += np.euler_gamma
                        _solution = spsolve( 
                                    (sp.eye(NBA*NBH*ZSHOCKS) - BETA*_Q), 
                                    _temp_u0 )
                        _temp_v.append( 
                                    _solution.reshape( (ZSHOCKS, NBA*NBH)).T )

            _distance = np.max( abs( np.array(self.dr) - np.array(_dr) ) )
            _iteration +=1
            self.v = _temp_v
            self.dr = _dr

            print("Iteration: {0:2d}; Distance: {1:d}".format(_iteration, _distance))
            if _iteration == MAXITERATIONS:
                print("Max number of iterations reached!")
    
    def save_to_csv(self):
        _size = len(pp_range)*len(o_range)*JOBS*NBA*NBH
        _df = pd.DataFrame(np.array(self.dr).reshape(_size, ZSHOCKS),
                                columns=z_shock_range)
        _df.to_csv('dr.csv')
        _df = pd.DataFrame(np.array(self.v).reshape(_size, ZSHOCKS),
                                columns=z_shock_range)
        _df.to_csv('v.csv')
        
    def load_from_csv(self):
        _df = pd.read_csv('dr.csv', index_col=0)
        self.dr = _df.as_matrix().reshape(len(pp_range)*len(o_range)*JOBS, 
                            NBA*NBH, ZSHOCKS)
        _df = pd.read_csv('v.csv', index_col=0)
        self.v = _df.as_matrix().reshape(len(pp_range)*len(o_range)*JOBS, 
                            NBA*NBH, ZSHOCKS)
    
    def show(self, i, j, k):
        plt.figure()
        plt.plot( self.h_grid, self.h_grid )
        plt.plot( self.h_grid,
                 self.h_grid[self.dr[i].
                 reshape(NBH, NBA, ZSHOCKS)[:, j, k] // NBA] )

        plt.figure()
        plt.plot( self.a_grid, self.a_grid )
        plt.plot( self.a_grid,
                 self.a_grid[self.dr[i].
                 reshape(NBH, NBA, ZSHOCKS)[j, :, k] % NBA] )

        plt.show()
        
        
# Moments of model (All):
PSI = 0.41                         # Labor supply elasticity
CHI = 0.36                         # Disutility of labor supply

m = Model(PSI, CHI, 0)
m.evaluate_model()
