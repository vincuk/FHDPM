from constants import o_range
from parameters import JOBS

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