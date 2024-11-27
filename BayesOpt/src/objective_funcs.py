import torch._dynamo
torch._dynamo.config.suppress_errors = True
import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.optimize import minimize

import torch

import matplotlib.pyplot as plt
FONTSIZE = 14
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}
import matplotlib as mp
mp.rc('font', **font)

# set usetex = False if LaTex is not
# available on your system or if the
# rendering is too slow
mp.rc('text', usetex=False)


import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal

#`source /cvmfs/sft.cern.ch/lcg/views/LCG_102/x86_64-centos7-gcc11-opt/setup.sh`
from glob import glob
from tqdm import tqdm
from configs import *

from BayesOpt_utils import *
def quadratic_form(point, values):
  y = 0
  scale = 1
  for x, (key, value) in zip(point, values.items()):
    y += scale * (x-value)**2
    scale *= 1.2
  return y

def toy_objective_func_one_min(aLund,
                     bLund,
                    rFactC,
                    rFactB,
                    aExtraSQuark,
                    aExtraDiquark,
                    # sigma,
                    # enhancedFraction,
                    # enhancedWidth,
                    # alphaSvalue,
                    # pTmin
                        ):
    # each minimum is a quadratic term
    # 3 minima
    point =         [aLund,
                     bLund,
                    rFactC,
                    rFactB,
                    aExtraSQuark,
                    aExtraDiquark,
                    # sigma,
                    # enhancedFraction,
                    # enhancedWidth,
                    # alphaSvalue,
                    # pTmin
                     ]

    y1 = quadratic_form(point, MONASH_DICT)


    result = y1

    return result# + np.random.normal(0,1)

def toy_objective_func_three_min(aLund,
                     bLund,
                    rFactC,
                    rFactB,
                    aExtraSQuark,
                    aExtraDiquark,
                    sigma,
                    enhancedFraction,
                    enhancedWidth,
                    alphaSvalue,
                    pTmin
                        ):
    # each minimum is a quadratic term
    # 3 minima
    point =         [aLund,
                     bLund,
                    rFactC,
                    rFactB,
                    aExtraSQuark,
                    aExtraDiquark,
                    sigma,
                    enhancedFraction,
                    enhancedWidth,
                    alphaSvalue,
                    pTmin
                     ]

    y1 = quadratic_form(point, MONASH_DICT)

    y2 = quadratic_form(point, POINT2)
    y3 = quadratic_form(point, POINT3)
    result = np.sqrt(y1 * (y2+1.0) * (y3+2.0))#/400
    return result# + np.random.normal(0,1)#random noise



def pythia_objective_func(aLund, 
                     bLund,
                    rFactC,
                    rFactB,
                    aExtraSQuark,
                    aExtraDiquark,
                    sigma,
                    enhancedFraction,
                    enhancedWidth,
                    # ProbStoUD,
                    # probQQtoQ,
                    # probSQtoQQ,
                    # ProbQQ1toQQ0,
                    alphaSvalue,
                    pTmin
                    ):
    
    # step 1: write .cmnd file 
    make_pythia_card(aLund, 
                     bLund,
                    rFactC,
                    rFactB,
                    aExtraSQuark,
                    aExtraDiquark,
                    sigma,
                    # enhancedFraction,
                    # enhancedWidth,
                    # ProbStoUD,
                    # probQQtoQ,
                    # probSQtoQQ,
                    # ProbQQ1toQQ0,
                    # alphaSvalue,
                    # pTmin
                    )
    #step 2 run main42 and rivet
    main42_path = os.path.join(BAYESOPT_BASE, 'BayesOp', 'src', 'main42')
    BO_card_path = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'BO_Cards', 'ALEPH_1996_S3486095_BO_card.cmnd')
    temp_path = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'temp')
    ALEPH_YODAS_path = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'data', 'ALEPH_YODAS_BayesOpt')

    os.system(f"""./{main42_path} {BO_card_path} {temp_path}/ALEPH_1996_S3486095_card.fifo
    rivet -o ALEPH_1996_S3486095_hist_0.yoda -a ALEPH_1996_S3486095 {temp_path}/ALEPH_1996_S3486095_card.fifo

    rm {temp_path}/ALEPH_1996_S3486095_card.fifo
    mv ALEPH_1996_S3486095_hist_0.yoda {ALEPH_YODAS_path}""")
    

    #step 3: get generated yoda file histograms in the form of dataframes
    dfdata, dfsims, generated_indices = get_data()
    print('DATA DATAFRAME')
    print(dfdata['/REF/ALEPH_1996_S3486095/d01-x01-y01'].head())
    print('FIRST SIM DATAFRAME')
    print(dfsims[generated_indices[0]]['/ALEPH_1996_S3486095/d01-x01-y01'].head())

    #step 4: fileter histograms based on our criteria
    data_keys, mc_keys = get_hist_names(dfdata)

    filtered_data_keys, filtered_mc_keys = filter_keys(dfdata, dfsims, data_keys, mc_keys)

    #step 4.5: take out bad histograms
    reduced_data_keys, reduced_mc_keys = reduce_filtered_keys(filtered_data_keys, filtered_mc_keys)
    print('reduced_data_keys, reduced_mc_keys', reduced_data_keys, reduced_mc_keys)
    reduced_data_keys, reduced_mc_keys = reduced_data_keys[:7], reduced_mc_keys[:7]
    
    #step 5: get test statistic at each point
    X0 = {}
    for ii, gen_ind in enumerate(generated_indices):
        # X0.append(test_statistic(filtered_data_keys,filtered_mc_keys, dfdata, dfsims[gen_ind], which = 0))
        # try:
        #     X0.append(test_statistic(filtered_data_keys,filtered_mc_keys, dfdata, dfsims[ii], which = 0))
        try:
            X0[gen_ind] = test_statistic(reduced_data_keys,
                                         reduced_mc_keys, 
                                         dfdata, 
                                         dfsims[gen_ind], 
                                         which = 0)
        except Exception:
            print('test statistic error in file index: ', gen_ind)
            
            
    objective_func = X0[0]
        
    os.system(f"rm {ALEPH_YODAS_path}/ALEPH_1996_S3486095_hist_0.yoda")
        
    print(f"objective function = {objective_func}")
    return objective_func

if __name__ == '__main__':
    pythia_objective_func(aLund=0.1, 
                     bLund=0.1,
                    rFactC=0.1,
                    rFactB=0.1,
                    aExtraSQuark=0.3,
                    aExtraDiquark=0.4,
                    sigma=0.5)