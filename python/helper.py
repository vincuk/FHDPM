import math
from constants import *
from parameters import *

def map_to_index(pp, j, o):
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
    return pp*( len(o_range)*JOBS ) + j*len(o_range) + _o

def wage(j, pp, h_old, l, o, z_shock):
    '''
    Earnings function
    
    :param int j: current job type
    :param int pp: 1 = receiving PP, 0 = not receiving PP
    :param float h_old: past period cumulative hours worked
    :param float l: labor supply
    :param float z_shock: shock
    '''
    if l <= 0:
        _log_labor = 0
    else:
        _log_labor = math.log(AUGMH * l)
    if o:
        _o = 1
    else:
        _o = 0
    return math.exp(
                WAGECONSTANT + 
                ALPHA*pp + 
                ZETA[j] + 
                GAMMA[0]*_log_labor +
                GAMMA[1]*math.log(h_old) + 
                GAMMA[2]*math.log(h_old)*pp + 
                XI*_o +
                z_shock
                )

def consumption(a_prime, a, w):
    '''
    Consumption
    
    :param float a_prime: next period assets
    :param float a: current peruod assets
    :param float w: wage
    '''
    return (1 + R)*a - a_prime + (1 - TAU)*w
                
def labor(h_prime, h):
    '''
    Labor supply
    
    :param float h_prime: next period cumulative hours worked
    :param float h: current peruod cumulative hours worked
    '''
    return (h_prime - h) / AUGMH
    
def utility(c, l):
    '''
    Utility function

    :param float c: consumption
    :param float l: labor supply
    '''
    return CHI*(c**(1-IOTA))/(1-IOTA) - (1 - CHI)*(l**(1+PSI))/(1+PSI)