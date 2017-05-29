import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from constants import *
from parameters import *
from helper import *

class Model:
    '''
    Model
    '''
        
    util = []
    dr = np.zeros( (NBA*NBH, ZSHOCKS), dtype=int )
    v = np.zeros( (NBA*NBH, ZSHOCKS), dtype=float )
    v = [v] * ( len(pp_range)*len(o_range)*JOBS )
    dr = [dr] * ( len(pp_range)*len(o_range)*JOBS )

    def __init__(self):
        
        self.a_grid = np.linspace(AMIN, AMAX, NBA)
        self.h_grid = np.linspace(HMIN, HMAX, NBH)

    def wage(self, j, pp, h_prime, h, o, z_shock):
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
            _h_old = HMIN + DELTAH*(2*h - h_prime)
        _labor =self.labor(h_prime, h)
        return wage( j, pp, _h_old, _labor, o, z_shock_range[z_shock] )
    
    def consumption(self, j, pp, h_prime, h, a_prime, a, o, z_shock):
        '''
        Consumption on grid
    
        :param int j: current job type
        :param int pp: 1 = receiving PP, 0 = not receiving PP
        :param int h_prime: index of the next period cumulative hours worked
        :param int h: index of the current peruod cumulative hours worked
        :param float a_prime: index of the next period assets
        :param float a: index of the current peruod assets
        :param int z_shock: z-shock index
        '''
        _w = self.wage(j, pp, h_prime, h, o, z_shock)
        return consumption( AMIN + DELTAA*a_prime, AMIN + DELTAA*a, _w )
                
    def labor(self, h_prime, h):
        '''
        Labor supply on grid
    
        :param int h_prime: index of the next period cumulative hours worked
        :param int h: index of the current peruod cumulative hours worked
        '''
        return DELTAH * labor(h_prime, h)
        
    def utility(self, c, l):
        '''
        Utility function on grid

        :param float c: consumption on grid
        :param float l: labor supply on grid
        '''
        return utility(c, l)
        
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
                                    if h_end - h_start > AUGMH/DELTAH:
                                        continue
                                    for a_end in range(a_start, NBA):
                                        if a_end - a_start > AUGMA + 1:
                                            continue
                                        _c = self.consumption(j, pp, 
                                                            h_end, h_start, 
                                                            a_end, a_start,
                                                            o, z_shock);
                                        if _c < 0:
                                            continue
                                        _l = self.labor(h_end, h_start)
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
    
    def show(self, i, j):
        plt.figure()
        _h_grid = self.h_grid / 1000
        plt.plot( _h_grid, _h_grid )
        plt.plot( _h_grid,
                 _h_grid[self.dr[i].
                 reshape(NBH,NBA,ZSHOCKS)[:, -1, j] // NBA] )

        plt.figure()
        plt.plot( self.a_grid, self.a_grid )
        plt.plot( self.a_grid,
                 self.a_grid[self.dr[i].
                 reshape(NBH,NBA,ZSHOCKS)[-1, :, j] % NBA] )

        plt.show()