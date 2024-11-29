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
BAYESOPT_BASE=os.environ['BAYESOPT_BASE']


def main():
    object_func = pythia_objective_func

    param_names = list(PARAM_DICT.keys())
    num_params =len(PARAM_DICT)
    print(f'num_params={num_params}')
    num_train_points = NUM_TRAIN_POINTS
    MAKE_TRAIN_DATASET = False
    if MAKE_TRAIN_DATASET:
        train_df_new = make_train_dataset(PARAM_DICT=PARAM_DICT, 
                                      points=num_train_points, 
                                      true_objective_func=object_func,
                                      save_data=True)
        print(train_df_new.head())

    else:
        train_df_new_path = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'data', 'pythia_objective_func_N_PYTHIA_EVENTS_250000_gp_train_data.csv')
        train_df_new = pd.read_csv(train_df_new_path)
        print(train_df_new.head())


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
                    retrain_gp=False,
                    print_=False,
                   save_model=True,
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