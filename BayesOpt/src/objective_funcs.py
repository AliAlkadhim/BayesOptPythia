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
