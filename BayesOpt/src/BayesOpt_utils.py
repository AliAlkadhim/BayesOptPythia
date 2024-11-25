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


def make_train_dataset(PARAM_DICT, points,true_objective_func,  ONLY_MONASH_TRAIN=False):
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
    return df

def print_parameters(model):
    state_dict = model.state_dict()
    for param_name, param_tensor in state_dict.items():
        print(param_name, param_tensor)




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
                       save_model=False,
                        OPTIMIZE_ACQ=False,
                        suggest_monash_point=False,
                        n_optimize_acq_iter=10,
                        n_restarts=25,
                        minimize_method='SLSQP',
                        jac=None):
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

        print(f'iteration {iteration} next_x = {next_x}, next_y = {next_y}')
        train_y = torch.cat([train_y, next_y])

        model.set_train_data(inputs=train_x, targets=train_y, strict=False)

        # retrain model with updated data
        if retrain_gp:
            model.train()
            model.likelihood.train()
            train_model(model, train_x, train_y, 5, print_=print_)

    train_size=train_x.shape[0]
    if save_model:
        path = f'models/GPytorch_all_params_model_Niter_{n_iterations}_trainsize_{train_size}_acq_{acquisition}.pth'
        torch.save(model.state_dict(), path)

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


