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

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel='RBF', heteroscedastic=False):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        if kernel == 'RBF':
          self.covar_module = ScaleKernel(RBFKernel(
              ard_num_dims=train_x.size(-1),
              # lengthscale_prior=gpytorch.priors.LogNormalPrior(loc= torch.log(torch.tensor([train_x.size(-1)/2])), scale=1)
          ))
        elif kernel == 'Matern':
          self.covar_module = ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=train_x.size(-1),
                                                                    #  lengthscale_prior=gpytorch.priors.LogNormalPrior(loc= torch.log(torch.tensor([train_x.size(-1)/2])), scale=1)
                                                                    )
          )
        elif kernel == 'RQ':
          self.covar_module = ScaleKernel(gpytorch.kernels.RQKernel(
              ard_num_dims=train_x.size(-1)
          ))
        elif kernel == 'Cosine':
          self.covar_module = ScaleKernel(gpytorch.kernels.CosineKernel(
              ard_num_dims=train_x.size(-1)
          ))

        elif kernel == 'Polynomial':
          self.covar_module = ScaleKernel(gpytorch.kernels.PolynomialKernel(power=5,
              ard_num_dims=train_x.size(-1)
          ))
        elif kernel == 'SpectralMixture':
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4,
              ard_num_dims=train_x.size(-1)
          )
            self.covar_module.initialize_from_data(train_x, train_y)

        self.heteroscedastic=heteroscedastic
        if heteroscedastic:
          self.noise_model = gpytorch.means.LinearMean(input_size=train_x.size(-1))


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        if self.heteroscedastic:
          noise = self.noise_model(x).exp().diag()
        else:
          noise=0
        return MultivariateNormal(mean_x, covar_x + noise)

    def predict(self, train_x):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad():
            pred = self(train_x)
        return self.likelihood(pred)

