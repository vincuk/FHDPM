import pandas as pd
import datetime

from python.model import Model
from python.parameters import AGENTS
from python.agents import simulate_agent, set_initial


if __name__ == '__main__':
    _key = datetime.datetime.now().strftime('%d-%m_%H-%M')
    _h = 500
    _a = 1
    _group = 0
    m = Model(_group)
    m.silent = False
    m.NBH = 900
    m.HMIN = _h
    m.HMAX = m.HMIN + m.AUGMH*15
    m.NBA = 10
    m.AMIN = _a
    m.AMAX = m.AMIN + 100000
    m.updade_parameters()
    m.evaluate_model()
    m.save_to_csv(_key)
    # m.load_from_csv("31-07_18-59")
    
    if m.TAU > 0.7:
        _filename = "data/results_" + _key + "_bigtau.csv"  
    else:
        _filename = "data/results_" + _key + ".csv"  
    
    for agent in range(1, AGENTS + 1):
        initials = set_initial(_a, _h, random = False)
        res_1 = simulate_agent(m, agent, initials)
        
        df1 = pd.DataFrame(res_1[:-1,0:4], dtype = 'int',
                            columns = ['Agent','Age','Job','PP'])
        df2 = pd.DataFrame(res_1[:-1,4:9], dtype = 'float',
                            columns = ['log_h_cum','l','c','log_w','shocks']) 
        df = pd.concat([df1,df2], axis = 1)
        if agent == 1:
            _mode = 'w'
            _header = True
        else:
            _mode = 'a'
            _header = False                   
        df.to_csv(_filename, index = False, header = _header, mode = _mode)