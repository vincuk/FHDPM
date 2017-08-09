import numpy as np
import scipy.optimize as opt
import pandas as pd
import datetime

from python.model import Model
# from python.rouwen import rouwen
from python.agents import integrate_agents



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

def data_moments(periods):
    min_per, max_per = periods
    _df = pd.read_csv("data/full_set_moments.csv", index_col=0)
    output = np.concatenate(
                (_df[_df.pp == 0]['llabinc'].as_matrix()[min_per:max_per],
                _df[_df.pp == 1]['llabinc'].as_matrix()[min_per:max_per],
                _df[_df.pp == 0]['lcumul_hours'].as_matrix()[min_per:max_per],
                _df[_df.pp == 1]['lcumul_hours'].as_matrix()[min_per:max_per])
                # _df[_df.pp == 0]['share_hours'].as_matrix()[min_per:max_per]/2,
                # _df[_df.pp == 1]['share_hours'].as_matrix()[min_per:max_per]/2)
             )                 
    return output

def sim_moments(params, *args):
    global initials 
    # , {9:1.2f}
    key = str('{0:1.2f}, {1:1.2f}, {2:1.2f}, {3:1.2f}, {4:1.2f}, ').format(
                params.tolist()[0], params.tolist()[1], params.tolist()[2], 
                params.tolist()[3], params.tolist()[4]) + \
          str('{0:1.2f}, {1:1.2f}, {2:1.2f}, {3:1.2f}').format(
                params.tolist()[5], params.tolist()[6], params.tolist()[7],
                params.tolist()[8])   
    print("params: ", key)
    _f.write("params: " + key + "\n")
    if key not in list(history.keys()):
        min_per, max_per = args[0]
        _h = 500
        _a = 1
        _llabinc0 = np.array([])
        _lcumul_hours0 = np.array([])
        _share_hours0 = np.array([])
        _llabinc1 = np.array([])
        _lcumul_hours1 = np.array([])
        _share_hours1 = np.array([])
        
        m = Model(0)
        m.NBH = 900
        m.HMIN = _h
        m.HMAX = m.HMIN + m.AUGMH*15.0
        m.NBA = 3
        m.AMIN = _a
        m.AMAX = m.AMIN + 3
        m.WAGECONSTANT = params[0]*10.0
        m.ALPHA = params[1]
        m.GAMMA[0] = params[2]
        m.GAMMA[1] = params[3]/10.0
        m.GAMMA[2] = params[4]/10.0
        m.XI = params[5]/10.0
        m.PSI = params[6]
        m.CHI[0] = params[7]
        m.CHI[1] = params[8]
        m.updade_parameters()
        m.evaluate_model()
            
        (res0, res1) = integrate_agents(m, _a, _h, random = False)
        
        _llabinc0 = np.append(_llabinc0, res0[:-1, 7])
        _llabinc1 = np.append(_llabinc1, res1[:-1, 7])
        _lcumul_hours0 = np.append(_lcumul_hours0, res0[:-1, 4])
        _lcumul_hours1 = np.append(_lcumul_hours1, res1[:-1, 4])   
        _share_hours0 = np.append(_share_hours0, res0[:-1, 5])
        _share_hours1 = np.append(_share_hours1, res1[:-1, 5])  
        
        history[key] = np.concatenate(
                        (_llabinc0[min_per:max_per], 
                        _llabinc1[min_per:max_per], 
                        _lcumul_hours0[min_per:max_per], 
                        _lcumul_hours1[min_per:max_per])
                        #_share_hours0[min_per:max_per]/2, 
                        #_share_hours1[min_per:max_per]/2)
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
    _f = open("data/SMM_log_" + datetime.datetime.now().strftime('%d-%m_%H-%M') + ".txt", 'w')

    params_init = np.array([0.68, -0.15, 0.65, 0.2, 0.3, 0.4, 0.4, 0.5, 0.6]) 
    periods = (1, 25)
 
    # criterion(params_init, periods)

    results = opt.minimize(criterion, params_init, args=(periods),
                          method='BFGS', 
                          options={'disp': True, 'maxiter' :10, 
                          'eps': 1e-2, 'gtol': 1e-2})
                          
    # results = opt.minimize(criterion, params_init, args=(periods),
    #                       method=custmin, options=dict(stepsize=0.01))
    
    _f.write('WAGECONSTANT = {0:1.2f}'.format(results.x[0]*10.0) + "\n")
    _f.write('ALPHA = {0:1.4f}'.format(results.x[1]) + "\n")
    _f.write('GAMMA[0] = {0:1.2f}'.format(results.x[2]) + "\n")
    _f.write('GAMMA[1] = {0:1.2f}'.format(results.x[3]/10.0) + "\n")
    _f.write('GAMMA[2] = {0:1.3f}'.format(results.x[4]/10.0) + "\n")
    _f.write('XI = {0:1.3f}'.format(results.x[5]/10.0) + "\n")
    _f.write('PSI 1 = {0:1.2f}'.format(results.x[6]) + "\n")
    _f.write('CHI 1 = {0:1.3f}'.format(results.x[7]) + "\n")
    _f.write('CHI 2 = {0:1.3f}'.format(results.x[8]) + "\n")
    _f.close()