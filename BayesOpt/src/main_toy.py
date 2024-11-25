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
from objective_funcs import *
from acquisition_funcs import *
from models import * 
from plotting import *

def main():
    object_func = toy_objective_func_one_min

    param_names = list(PARAM_DICT.keys())
    num_params =len(PARAM_DICT)
    print(f'num_params={num_params}')
    num_train_points = num_params *6
    train_df_new = make_train_dataset(PARAM_DICT=PARAM_DICT, points=num_train_points, true_objective_func=object_func)
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
    model = train_model(model=model, train_x=train_x, train_y=train_y,  n_epochs=N_TRAIN_EPOCHS,print_=False)

    # plot_all(model=model)


    iterations, true_objective_funcs = BayesOpt_all_params(true_objective_func=object_func,
                                                                   model=model,
                                                                   optimize_acq_method=OPTIMIZE_ACQ_METHOD,
                    train_x=train_x,
                    train_y=train_y,
                    n_iterations=N_BO_ITERATIONS,
                    acquisition = 'EI',
                    retrain_gp=False,
                    print_=False,
                   save_model=False,
                    OPTIMIZE_ACQ=True,
                    suggest_monash_point=False,
                    n_optimize_acq_iter=N_OPTIMIZE_ACQ_ITER,
                    n_restarts=N_RESTARTS,
                    minimize_method='Nelder-Mead',
                    jac=None)
    
    best_parameters, best_f = get_observed_best_parameters(model)
    print(f'best_parameters={best_parameters}')
    print(f'best_f={best_f}')
    plot_all(model=model)
    

if __name__=="__main__":
    main()