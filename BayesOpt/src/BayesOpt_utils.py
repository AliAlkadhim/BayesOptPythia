from datetime import datetime
import os

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
from models import *
from acquisition_funcs import *
from objective_funcs import *
from optimize_acquisition import *
from shared_utils import *
from pythia_SBI_utils import *
from yoda2numpy_BayesOpt import *

BAYESOPT_BASE=os.environ['BAYESOPT_BASE']





def make_x_candidates(PARAM_DICT, size):
    """
    Example: x_candidates = make_x_candidates(PARAM_DICT,2)
    """
    full_matrix = np.empty((size, len(PARAM_DICT)))

    for row in range(size):
        result=[]
        for key, val in PARAM_DICT.items():
            param_name, range_ = key, val
            param_postfix = param_name.split(':')[1]
            uniform_size_1 = Sample_param(param=param_postfix, size=1).uniform().item()
            result.append(uniform_size_1)
        full_matrix[row,:] = result

    result = full_matrix

    # print(result)
    return torch.tensor(result, requires_grad=True)


def make_multidim_xstar(model, param,size):
    """
    This is used for sampling the parameters for plotting
    """
    train_x = model.train_inputs[0].numpy()
    train_y = model.train_targets.numpy()
    train_x = train_x[train_y.argmin()]


    x_star0 = Sample_param(param,size).linspace()
    empty = np.ones((size, len(PARAM_DICT)))
    for param_indx in np.arange(len(PARAM_DICT)):
        empty[:,param_indx] = train_x[param_indx]
    # empty = make_x_candidates(PARAM_DICT, size)
    param_prefix = get_param_prefix(param)
    full_param_name = param_prefix + ':' + param
    param_index = param_names.index(full_param_name)
    empty[:,param_index] = x_star0
    # print(empty)
    return torch.tensor(empty)


def make_train_dataset(PARAM_DICT, points,true_objective_func, save_data=True):
    param_names = list(PARAM_DICT.keys())
    column_names = param_names + ['chi2']

    rows = []
    for _ in range(points):
        row = []
        for param_name, range_ in PARAM_DICT.items():
            param_postfix = param_name.split(':')[1]
            uniform_size_1 = Sample_param(param=param_postfix, size=1).uniform().item()
            row.append(uniform_size_1)

        chi2 = true_objective_func(*row)
        row.append(chi2)
        rows.append(row)

    df = pd.DataFrame(rows, columns=column_names)
    if save_data:
        df.to_csv(os.path.join(BAYESOPT_BASE, 'BayesOpt', 'data', f'{true_objective_func.__name__}_train_data.csv'))
    return df

def print_parameters(model):
    state_dict = model.state_dict()
    for param_name, param_tensor in state_dict.items():
        print(param_name, param_tensor)

def configs_df():
    configs_dict = {
        'N_BO_ITERATIONS': N_BO_ITERATIONS,
        'N_PARAMS': num_params,
        'N_OPTIMIZE_ACQ_ITER': N_OPTIMIZE_ACQ_ITER,
        'N_RESTARTS': N_RESTARTS,
        'KERNEL': KERNEL,
        'OPTIMIZE_ACQ_METHOD': OPTIMIZE_ACQ_METHOD,
    }
    df = pd.DataFrame([configs_dict])
    return df

def model_history_df(model):
    param_names = list(PARAM_DICT.keys())
    column_names = param_names + ['chi2']
    train_x = model.train_inputs[0].numpy()
    print(f'MODEL train_x.shape={train_x.shape}')
    train_y = model.train_targets.numpy().reshape(-1,1)
    print(f'MODEL train_y.shape={train_y.shape}')
    data = np.hstack([train_x, train_y])
    df = pd.DataFrame(data, columns=column_names)
    return df


def directory_name():
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f'Tune_{time}'
    return dir_name

def train_model(model, train_x, train_y, n_epochs, print_=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model )

    model.train()
    model.likelihood.train()

    eps=2e-6
    loss_prev = torch.tensor([np.inf])
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = model(train_x)  # Use model() instead of model.predict()
        loss = -mll(output, train_y)

        if print_:
            print(f'Epoch {epoch + 1}/{n_epochs} - Loss: {loss.item():.3f}   '
                  f'lengthscale: {model.covar_module.base_kernel.lengthscale.tolist()[0]}   '
                  f'noise: {model.likelihood.noise.item():.3f}')



        loss.backward()
        optimizer.step()
        if torch.abs(loss_prev-loss) < eps:
          break
        loss_prev = loss

    model.eval()
    # model.likelihood.eval()
    return model




def BayesOpt_all_params(true_objective_func,
                        model,
                        optimize_acq_method,
                        train_x,
                        train_y,
                        n_iterations,
                        acquisition = 'EI',
                        retrain_gp=False,
                        print_=False,
                       save_model=True,
                        OPTIMIZE_ACQ=False,
                        suggest_monash_point=False,
                        n_optimize_acq_iter=10,
                        n_restarts=N_RESTARTS,
                        minimize_method='SLSQP',
                        jac=None,
                        save_output=True,
                        kappa=KAPPA):
    # Use the Adam optimizer

    model.eval()
    model.likelihood.eval()

    ONLY_MONASH_TRAIN=False

    if suggest_monash_point:
      Monash_candidate = torch.tensor(list(MONASH_DICT.values()))
      next_y = true_objective_func(*Monash_candidate)
      print(f'objective function at the Monash point = {next_y}')
      if ONLY_MONASH_TRAIN:
          train_x = torch.cat([train_x.unsqueeze(0), Monash_candidate.unsqueeze(0)])
      else:
          train_x = torch.cat([train_x, Monash_candidate.unsqueeze(0)])
      next_y = torch.tensor([next_y])
      if ONLY_MONASH_TRAIN:
          train_y = torch.cat([train_y.unsqueeze(0), next_y])
      else:
          train_y = torch.cat([train_y, next_y])
      model.set_train_data(inputs=train_x, targets=train_y, strict=False)
    ######################################################################
    iterations=[]
    true_objecctive_funcs=[]
    for iteration in range(n_iterations):
        iterations.append(iteration)
        if OPTIMIZE_ACQ:
          x_candidates = make_sobol_candidates(PARAM_DICT,1)
          def acq_(x):
            if acquisition == 'EI':
              return - expected_improvement(model, train_y, x)
            elif acquisition == 'UCB':
              return - ucb(model, x)

          if acquisition == 'EI':
              acq = expected_improvement(model, train_y, x_candidates)
          elif acquisition == 'UCB':
              acq = ucb(model, x_candidates)

          # next_x = optimize_acq_2(acquisition_func=expected_improvement, x_candidates=x_candidates)

          # next_x = optimize_acq_3(acq_=acq_, num_optimize_iterations= 1000, x_init=x_candidates)

          
          if optimize_acq_method == 'scipy':
            def acq_numpy(x):
              return - expected_improvement_numpy(model, train_y, x).detach().numpy()
            
            next_x = minimize_acq_scipy(acq_numpy, n_restarts=n_restarts, minimize_method=minimize_method,jac = jac)
          elif optimize_acq_method=='Adam_restarts_clip_bounds':
            next_x = optimize_acq(model=model, acquisition_func=expected_improvement, n_optimize_acq_iter=N_OPTIMIZE_ACQ_ITER, n_restarts=N_RESTARTS)

          elif optimize_acq_method=='Adam_no_clip_bounds':
            next_x = optimize_acq_2(model=model, acquisition_func=expected_improvement, x_candidates=x_candidates, n_optimize_acq_iter=N_OPTIMIZE_ACQ_ITER)

          elif optimize_acq_method=='GD':
            def acq_(x):
                # Ensure x is a tensor with grad enabled if it's not already
                # x_tensor = x if x.requires_grad else torch.tensor(x, dtype=torch.float64, requires_grad=True)
                return -expected_improvement(model=model, observed_y=train_y, candidate_set=x)

            next_x = optimize_acq_3(acq_=acq_, num_optimize_iterations= N_OPTIMIZE_ACQ_ITER, x_init=x_candidates)
         
         
          # next_x=next_x.detach()

        else:
          #if no aquisition function is defined, just take the literal argmax of the acquisition function
          x_candidates = make_sobol_candidates(PARAM_DICT,1000)
        # x_candidates = torch.cat([Monash_candidate.unsqueeze(0),x_candidates])
          if acquisition == 'EI':
              acq = expected_improvement(model, train_y, x_candidates)
          elif acquisition == 'UCB':
              acq = ucb(model, x_candidates)
          x_candidates=x_candidates.detach()
          acq_argmax = acq.argmax()
          next_x = x_candidates[acq_argmax]

        next_y = true_objective_func(*next_x)
        true_objecctive_funcs.append(next_y)

        if ONLY_MONASH_TRAIN:
            train_x = torch.cat([train_x, next_x.unsqueeze(0)])
        else:
            train_x = torch.cat([train_x, next_x.unsqueeze(0)])
        next_y = torch.tensor([next_y])

        print(f'iteration {iteration}, next_x = {next_x}, next_y = {next_y}, acq = {acq}')
        train_y = torch.cat([train_y, next_y])

        model.set_train_data(inputs=train_x, targets=train_y, strict=False)

        # retrain model with updated data
        if retrain_gp:
            model.train()
            model.likelihood.train()
            train_model(model, train_x, train_y, 5, print_=print_)
    
        if acq < kappa:
            print(f'STOPPING BO LOOP AT ITERATION {iteration}')
            break
    ####################################################################
    # End BO loop
    train_size=train_x.shape[0]
    # we want to always save the model for every run because it has the whole history and data

    if save_model:
        dir_name = directory_name()

        dir_path = make_output_dirname(dir_name)
        path = os.path.join(dir_path, 'model.pth')

        torch.save(model.state_dict(), path)

    if save_output:
        configs = configs_df()
        
        history_df = model_history_df(model)

        dir_name = directory_name()
        dir_path = make_output_dirname(dir_name)

        df_path = os.path.join(dir_path, 'history.csv')
        
        history_df.to_csv(df_path, index=False)

        configs_df_path = os.path.join(dir_path, 'configs.csv')
        configs.to_csv(configs_df_path, index=False)

    return iterations, true_objecctive_funcs


def get_observed_best_parameters(model):
    train_x = model.train_inputs[0].numpy()
    train_y = model.train_targets.numpy()
    best_f = train_y.min()
    observed_min = train_x[train_y.argmin()]
    param_names = list(PARAM_DICT.keys())
    param_names = [param_name.split(':')[1] for param_name in param_names]
    best_params_dict = {k: v for k, v in zip(param_names, observed_min)}
    return best_params_dict, best_f


def make_output_dirname(dir_name):
    output_path = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'output', dir_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path

def make_pythia_card(aLund, 
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
                    ):
    
    BO_Cards_dir = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'BO_Cards')
    if not os.path.exists(BO_Cards_dir):
        os.makedirs(BO_Cards_dir)

    filename = f"ALEPH_1996_S3486095_BO_card.cmnd"
    file_path = os.path.join(BO_Cards_dir, filename)
    with open(file_path,'w') as f:
        first_block="""Main:numberOfEvents = 3000          ! number of events to generate
Next:numberShowEvent = 0           ! suppress full listing of first events
# random seed
Random:setSeed = on
Random:seed= 0
! 2) Beam parameter settings.
Beams:idA = 11                ! first beam,  e- = 11
Beams:idB = -11                ! second beam, e+ = -11
Beams:eCM = 91.2               ! CM energy of collision
# Pythia 8 settings for LEP
# Hadronic decays including b quarks, with ISR photons switched off
WeakSingleBoson:ffbar2gmZ = on
23:onMode = off
23:onIfAny = 1 2 3 4 5
PDF:lepton = off
SpaceShower:QEDshowerByL = off\n\n"""
        f.write(first_block)
        # f.write(f"Random:seed={indx+1}")
        f.write(f"StringZ:aLund = {aLund}\n\n")
        f.write(f"StringZ:bLund = {bLund}\n\n")
        f.write(f"StringZ:rFactC = {rFactC}\n\n")
        f.write(f"StringZ:rFactB = {rFactB}\n\n")
        f.write(f"StringZ:aExtraSQuark = {aExtraSQuark}\n\n")
        f.write(f"StringZ:aExtraDiquark = {aExtraDiquark}\n\n")
        f.write(f"StringPT:sigma = {sigma}\n\n")
        # f.write(f"StringPT:enhancedFraction = {enhancedFraction}\n\n")
        # f.write(f"StringPT:enhancedWidth = {enhancedWidth}\n\n")
        # f.write(f"StringFlav:ProbStoUD = {ProbStoUD}\n\n")
        # f.write(f"StringFlav:probQQtoQ = {probQQtoQ}\n\n")
        # f.write(f"StringFlav:probSQtoQQ = {probSQtoQQ}\n\n")
        # f.write(f"StringFlav:ProbQQ1toQQ0 = {ProbQQ1toQQ0}\n\n")
        # f.write(f"TimeShower:alphaSvalue = {alphaSvalue}\n\n")
        # f.write(f"TimeShower:pTmin = {pTmin}\n\n")
        


def get_pbounds(PARAM_DICT):
    pbounds = {}
    for key, value in PARAM_DICT.items():
        p_name = key.split(':')[1]
        p_bound = tuple(value)
        pbounds[p_name] = p_bound
    return pbounds




def reduce_filtered_keys(filtered_data_keys, filtered_mc_keys):
    # Initialize empty list for the reduced keys
    reduced_data_keys = []
    reduced_mc_keys = []
    # List of histogram keys that need to be removed
    hists_to_remove = ['d35-x01-y01', 'd36-x01-y01', 'd39-x01-y01', 'd40-x01-y01']
    
    # Iterate over each data key
    for data_key in filtered_data_keys:
        # Add the key to reduced_data_keys only if it does not match any hist_to_remove
        if not any(hist_to_remove in str(data_key) for hist_to_remove in hists_to_remove):
            reduced_data_keys.append(data_key)

    for mc_key in filtered_mc_keys:
        # Add the key to reduced_data_keys only if it does not match any hist_to_remove
        if not any(hist_to_remove in str(mc_key) for hist_to_remove in hists_to_remove):
            reduced_mc_keys.append(mc_key)
            
        
    return reduced_data_keys, reduced_mc_keys
        




def make_pythia_valid_card(best_parameters):
    
    cards_dir = os.path.join(os.getcwd(), "BO_Cards")
    filename = f"ALEPH_1996_S3486095_BO_card_valid.cmnd"
    file_path = os.path.join(cards_dir, filename)
    with open(file_path,'w') as f:
        first_block="""Main:numberOfEvents = 300000          ! number of events to generate
Next:numberShowEvent = 0           ! suppress full listing of first events
# random seed
Random:setSeed = on
Random:seed= 0
! 2) Beam parameter settings.
Beams:idA = 11                ! first beam,  e- = 11
Beams:idB = -11                ! second beam, e+ = -11
Beams:eCM = 91.2               ! CM energy of collision
# Pythia 8 settings for LEP
# Hadronic decays including b quarks, with ISR photons switched off
WeakSingleBoson:ffbar2gmZ = on
23:onMode = off
23:onIfAny = 1 2 3 4 5
PDF:lepton = off
SpaceShower:QEDshowerByL = off\n\n"""
        f.write(first_block)
        # f.write(f"Random:seed={indx+1}")

        f.write("StringZ:aLund = {}\n\n".format(best_parameters["aLund"]))
        f.write("StringZ:bLund = {}\n\n".format(best_parameters["bLund"]))
        f.write("StringZ:rFactC = {}\n\n".format(best_parameters["rFactC"]))
        f.write("StringZ:rFactB = {}\n\n".format(best_parameters["rFactB"]))
        f.write("StringZ:aExtraSQuark = {}\n\n".format(best_parameters["aExtraSQuark"]))
        f.write("StringZ:aExtraDiquark = {}\n\n".format(best_parameters["aExtraDiquark"]))
        f.write("StringPT:sigma = {}\n\n".format(best_parameters["sigma"]))
        f.write("StringPT:enhancedFraction = {}\n\n".format(best_parameters["enhancedFraction"]))
        f.write("StringPT:enhancedWidth = {}\n\n".format(best_parameters["enhancedWidth"]))
        f.write("StringFlav:ProbStoUD = {}\n\n".format(best_parameters["ProbStoUD"]))
        f.write("StringFlav:probQQtoQ = {}\n\n".format(best_parameters["probQQtoQ"]))
        f.write("StringFlav:probSQtoQQ = {}\n\n".format(best_parameters["probSQtoQQ"]))
        f.write("StringFlav:ProbQQ1toQQ0 = {}\n\n".format(best_parameters["ProbQQ1toQQ0"]))
        f.write("TimeShower:alphaSvalue = {}\n\n".format(best_parameters["alphaSvalue"]))
        f.write("TimeShower:pTmin = {}\n\n".format(best_parameters["pTmin"]))

def run_valid_card(best_parameters):
    
    # step 1: write .cmnd file 
    make_pythia_valid_card(best_parameters)
    #step 2 run main42 and rivet
    os.system("""./main42 BO_Cards/ALEPH_1996_S3486095_BO_card_valid.cmnd /media/ali/DATA/TEMP/ALEPH_1996_S3486095_card_valid.fifo
    rivet -o ALEPH_1996_S3486095_hist_valid_0.yoda -a ALEPH_1996_S3486095 /media/ali/DATA/TEMP/ALEPH_1996_S3486095_card_valid.fifo

    rm /media/ali/DATA/TEMP/ALEPH_1996_S3486095_card_valid.fifo
    mv ALEPH_1996_S3486095_hist_valid_0.yoda ALEPH_YODAS_BayesOpt/""")


if __name__ == '__main__':
    make_pythia_card(aLund=0.1  , 
                     bLund=0.1,
                    rFactC=0.1,
                    rFactB=0.1,
                    aExtraSQuark=0.3,
                    aExtraDiquark=0.4,
                    sigma=0.5)