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

from shared_utils import *
from acquisition_funcs import *

# def optimize_acq(model,
#                  acquisition_func,
#                  n_optimize_acq_iter,
#                  n_restarts
#                  ):
#     train_y = model.train_targets
#     best_x = None
#     best_acq_value = float('inf')

#     lr = 0.01 if num_params <= 2 else 0.001

#     for _ in range(n_restarts):
#         x_candidates = make_sobol_candidates(PARAM_DICT, 1)
#         optimizer = torch.optim.Adam([x_candidates], lr=lr)
        
#         for i in range(n_optimize_acq_iter):
#             optimizer.zero_grad()
#             loss = -acquisition_func(model, train_y, x_candidates)
#             loss.backward()
#             optimizer.step()
            
#             if loss.item() < best_acq_value:
#                 best_acq_value = loss.item()
#                 best_x = clip_to_bounds(x_candidates.reshape(-1))

#     return best_x

def optimize_acq(model,
                 acquisition_func,
                 n_optimize_acq_iter,
                 n_restarts
                 ):
    train_y = model.train_targets
    best_x = None
    best_acq_value = float('inf')

    lr = 0.005
    
    for _ in range(n_restarts):
        # Generate multiple candidates and evaluate them individually
        x_candidates = make_sobol_candidates(PARAM_DICT, 1)
        # init_acq_values = torch.tensor([
        #     -acquisition_func(model, train_y, candidates[i:i+1]) 
        #     for i in range(candidates.shape[0])
        # ])
        
        # # Create a new leaf tensor for optimization
        # x_candidates = candidates[torch.argmin(init_acq_values)].clone().detach().reshape(1, -1)
        x_candidates.requires_grad_(True)
        
        optimizer = torch.optim.Adam([x_candidates], lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        for i in range(n_optimize_acq_iter):
            optimizer.zero_grad()
            loss = -acquisition_func(model, train_y, x_candidates)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([x_candidates], max_norm=1.0)
            optimizer.step()
            scheduler.step(loss)
            
            # Clip after optimization step
            with torch.no_grad():
                x_candidates.data = clip_to_bounds(x_candidates.reshape(-1)).reshape(1, -1)
            
            if loss.item() < best_acq_value:
                best_acq_value = loss.item()
                best_x = x_candidates.reshape(-1).detach().clone()

    return best_x

def optimize_acq_2(model, acquisition_func,
                   x_candidates,
                   n_optimize_acq_iter
                   ):
  """Adam with n_optimize_acq_iter (no clip to bounds)"""
  train_y = model.train_targets
  
  if num_params <= 2:
    lr = 0.1
  else:
    lr = 0.06
  optimizer = torch.optim.Adam([x_candidates],lr=lr)
  eps=1e-8
  loss_prev = torch.tensor([np.inf])
  for i in range(n_optimize_acq_iter):
    optimizer.zero_grad()
    loss= - acquisition_func(model, train_y, x_candidates)
    loss.requires_grad_(True)
    loss.backward()
    optimizer.step()
    if torch.abs(loss_prev-loss) < eps:
      break
    loss_prev = loss
  return x_candidates.detach().reshape(-1)



def optimize_acq_3(acq_, num_optimize_iterations, x_init):
  x_init=x_init.clone().detach().requires_grad_(True)
  # print(x_init)
  eps=1e-8
  learning_rate=torch.tensor([0.01])
  loss_prev = torch.tensor([np.inf])
  for i in range(num_optimize_iterations):
    loss= acq_(x_init)
    loss.requires_grad_(True)
    loss.backward()
    # print(x_init.grad)
    with torch.no_grad():
      x_init -= learning_rate * x_init.grad

    x_init.grad.zero_()
    if torch.abs(loss_prev-loss) < eps:
      break
    loss_prev = loss
  return x_init.reshape(-1)


def minimize_acq_scipy(acq, n_restarts, minimize_method='SLSQP', jac=None):
  # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for listing of available optimization methods
    best_x = None
    best_acq_value = float('inf')
    bounds = np.array(list(PARAM_DICT.values()))

    for _ in range(n_restarts):
      x0 =  make_sobol_candidates(PARAM_DICT,1).detach().numpy().reshape(-1)#x0 has to have one dimension, reshape inside EI_numpy
      # 'L-BFGS-B' 'SLSQP'
      if jac is None:
        res = minimize(acq, x0, bounds=bounds, method=minimize_method)
      else:
        res = minimize(acq, x0, bounds=bounds, method=minimize_method,jac=jac)
      if res.fun < best_acq_value:
          best_acq_value = res.fun
          best_x = res.x
    return torch.tensor(best_x)






def jac_acq_(x):
  x = torch.tensor(x,requires_grad=True)
  F = acq_(x)
  dFdX = torch.autograd.grad(outputs=F, inputs=x,
                               grad_outputs=torch.ones_like(F),
                               allow_unused=True,
                               #retain_graph=True,
                               create_graph=True)#[0]
  Y = dFdX.detach().numpy()
  if len(Y) == 1:
    Y = Y[0]
  return Y

def jac_acq_2(x):
    x_tensor = torch.tensor(x, requires_grad=True)
    # x_tensor = x_tensor.reshape(1, -1)

    F = acq_(x_tensor)
    F.backward()

    grad = x_tensor.grad
    return grad.numpy().flatten()