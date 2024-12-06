import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.optimize import minimize

import torch

import scienceplots
import matplotlib.pyplot as plt
import matplotlib as mp
plt.style.use(['science', 'notebook', 'grid'])

# Then set custom font configurations
FONTSIZE = 20
font = {
    'family': 'serif',
    'weight': 'normal',
    'size': FONTSIZE
}
mp.rc('font', **font)


import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal

#`source /cvmfs/sft.cern.ch/lcg/views/LCG_102/x86_64-centos7-gcc11-opt/setup.sh`
from glob import glob
from tqdm import tqdm
# from configs import *

from BayesOpt_utils import *
from objective_funcs import *
from acquisition_funcs import *
from models import * 
from plotting import *


PARAM_DICT = {
        'StringZ:aLund' : [0, 2.0],
        'StringZ:bLund': [0, 2.0],
        # 'StringZ:rFactC':[0.0, 2.0],
        # 'StringZ:rFactB': [0., 2.0],
        # 'StringZ:aExtraSQuark':[0.,2.],
        # 'StringZ:aExtraDiquark':[0.,2.],
        # 'StringPT:sigma':[0.,1.],
        # 'StringPT:enhancedFraction':[0.,1.],
        # 'StringPT:enhancedWidth':[1.0,10.0],
        'StringFlav:ProbStoUD':[0,1.0],
        'StringFlav:probQQtoQ':[0,1.0],
        # 'StringFlav:probSQtoQQ':[0,1.0],
        # 'StringFlav:ProbQQ1toQQ0':[0,1.0],
        'TimeShower:alphaSvalue':[0.06,0.25],
        'TimeShower:pTmin':[0.1,2.0]
}

MONASH_DICT = {
    "aLund" : 0.68, 
    "bLund" : 0.98,
    # "rFactC": 1.32,
    # "rFactB":0.855,
    # "aExtraSQuark": 0.0,
    # "aExtraDiquark":0.97,
    # "sigma":0.335,
    # "enhancedFraction":0.01,
    # "enhancedWidth":2.0,
    "ProbStoUD":0.217,
    "probQQtoQ":0.081,
    # "probSQtoQQ":0.915,
    # "ProbQQ1toQQ0": 0.0275,
    "alphaSvalue": 0.1365,
    "pTmin": 0.5
}

param_names = list(PARAM_DICT.keys())
num_params = len(param_names)
POINT2 = {key:0.5*value for key, value in MONASH_DICT.items()}
POINT3 = {key:1.5*value for key, value in MONASH_DICT.items()}


NUM_PYTHIA_EVENTS=250000
# NUM_PYTHIA_EVENTS=1000
#250000, 1000
KERNEL = 'Matern'
NUM_TRAIN_POINTS=25
N_TRAIN_EPOCHS=25
N_BO_ITERATIONS = 80# num_params * 30
#N_BO_ITERATIONS = 2
N_OPTIMIZE_ACQ_ITER = 50
N_RESTARTS = 10
# OPTIMIZE_ACQ_METHOD: ['GD', 'Adam_restarts_clip_bounds', 'Adam_no_clip_bounds', 'scipy']
OPTIMIZE_ACQ_METHOD = 'Adam_restarts_clip_bounds'
KAPPA=-1E-5

def main():
    object_func = toy_objective_func_one_min
    # object_func = toy_objective_func_three_min

    param_names = list(PARAM_DICT.keys())
    num_params =len(PARAM_DICT)
    print(f'num_params={num_params}')
    num_train_points = NUM_TRAIN_POINTS
    train_df_new = make_train_dataset(PARAM_DICT=PARAM_DICT, 
                                      points=num_train_points, 
                                      true_objective_func=object_func,
                                      save_data=True)
    print(train_df_new.head())

    MINIMA = [MONASH_DICT, POINT2, POINT3]
    for i in MINIMA:
        print(object_func(**i))


    train_x = train_df_new[param_names].to_numpy()
    train_y = train_df_new['chi2'].to_numpy()
    print(f'train_x.shape={train_x.shape}') 
    print(f'train_y.shape={train_y.shape}')


    # define model and likelihood
    train_x = torch.tensor(train_x).clone().detach()
    train_y = torch.tensor(train_y).clone().detach()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    #ExactGP can only handle Gaussian likelihoods


    model = GPModel(train_x = train_x, train_y=train_y, likelihood=likelihood, kernel=KERNEL, heteroscedastic=False).double()

    # print_parameters(model)
    model = train_model(model=model, 
                        train_x=train_x, 
                        train_y=train_y,  
                        n_epochs=N_TRAIN_EPOCHS,
                        print_=False,
                        plot_loss=False)

    # plot_all(model=model)


    iterations, true_objective_funcs, dir_name = BayesOpt_all_params(true_objective_func=object_func,
                                                                   model=model,
                                                                   optimize_acq_method=OPTIMIZE_ACQ_METHOD,
                    train_x=train_x,
                    train_y=train_y,
                    n_iterations=N_BO_ITERATIONS,
                    acquisition = 'EI',
                    retrain_gp=True,
                    print_=False,
                   save_model=False,
                    OPTIMIZE_ACQ=True,
                    suggest_monash_point=False,
                    n_optimize_acq_iter=N_OPTIMIZE_ACQ_ITER,
                    n_restarts=N_RESTARTS,
                    minimize_method='SLSQP',
                    jac=None,
                    save_output=True,
                        kappa=KAPPA, 
                        params='CONFIG')
    
    
    best_parameters, best_f = get_observed_best_parameters(model=model, true_objective_func=object_func, params='CONFIG')
    print(f'best_parameters={best_parameters}')
    print(f'best_f={best_f}')

    plot_all(model=model,dirname=dir_name, set_xy_lim=False, save_fig=True)
    

if __name__=="__main__":
    main()