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

def expected_improvement(model, observed_y, candidate_set):
    model.eval()
    model.likelihood.eval()
    # Don't do candidate_set = torch.tensor(candidate_set,requires_grad=True), it stops the computation graph!
    candidate_set = candidate_set.reshape(1,-1)

    with gpytorch.settings.fast_pred_var():
        observed_pred = model(candidate_set)  # Use model() instead of model.predict()
        best_f = observed_y.min()

        mean = observed_pred.mean
        sigma = observed_pred.variance.sqrt()

        xi=0.01

        z = ((best_f-xi) - mean) / sigma

        # Compute EI
        normal = torch.distributions.Normal(0, 1)
        
        ei = sigma * (z * normal.cdf(z) + normal.log_prob(z).exp())

        # Set EI to 0 where sigma is 0 
        ei = torch.where(sigma > 0, ei, torch.zeros_like(ei))
    # print('candidate set', candidate_set)
    # print('ei output', ei)
    return ei

def expected_improvement_numpy(model, observed_y, candidate_set):

    candidate_set = torch.tensor(candidate_set).reshape(1,-1)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = model(candidate_set)  # Use model() instead of model.predict()
        best_f = observed_y.min()

        mean = observed_pred.mean
        sigma = observed_pred.variance.sqrt()
        
        xi=0.01

        z = ((best_f-xi) - mean) / sigma

        # Compute EI
        normal = torch.distributions.Normal(0, 1)
        ei = sigma * (z * normal.cdf(z) + normal.log_prob(z).exp())

        # Set EI to 0 where sigma is 0 (to avoid NaN)
        ei = torch.where(sigma > 0, ei, torch.zeros_like(ei))

    return ei

def ucb(model, candidate_set, kappa=1.5):
    with gpytorch.settings.fast_pred_var():
        pred = model(candidate_set)
    return pred.mean + kappa * pred.stddev
