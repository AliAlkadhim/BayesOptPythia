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
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.distributions import MultivariateNormal

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import init_notebook_plotting, render

from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
import botorch
botorch.settings.debug(True)


#`source /cvmfs/sft.cern.ch/lcg/views/LCG_102/x86_64-centos7-gcc11-opt/setup.sh`
from glob import glob
from tqdm import tqdm

from typing import Optional
from torch import Tensor


from BayesOpt_utils import *
from objective_funcs import *
from acquisition_funcs import *
from models import * 
from plotting import *
BAYESOPT_BASE=os.environ['BAYESOPT_BASE']


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


N_TRAIN_POINTS = 25

# n_bo_iterations_l = [60, 80, 90]
n_bo_iterations_l = [120]
class SimpleCustomGP(ExactGP, GPyTorchModel):

    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y, train_Yvar: Optional[Tensor] = None):
        # NOTE: This ignores train_Yvar and uses inferred noise instead.
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        # self.covar_module = ScaleKernel(
        #     base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
        # )
        self.covar_module = ScaleKernel(
            base_kernel=MaternKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    

def main():
    for n_bo_iterations in n_bo_iterations_l:
        ax_model = BoTorchModel(
            surrogate=Surrogate(
                # The model class to use
                botorch_model_class=SimpleCustomGP,
                # Optional, MLL class with which to optimize model parameters
                # mll_class=ExactMarginalLogLikelihood,
                # Optional, dictionary of keyword arguments to model constructor
                # model_options={}
            ),
            # Optional, acquisition function class to use - see custom acquisition tutorial
            # botorch_acqf_class=qExpectedImprovement,
        )


        gs = GenerationStrategy(
            steps=[
                # Quasi-random initialization step
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=N_TRAIN_POINTS,  # How many trials should be produced from this generation step
                ),
                # Bayesian optimization step using the custom acquisition function
                GenerationStep(
                    model=Models.BOTORCH_MODULAR,
                    num_trials=n_bo_iterations,  # No limitation on how many trials should be produced from this step
                    # For `BOTORCH_MODULAR`, we pass in kwargs to specify what surrogate or acquisition function to use.
                    model_kwargs={
                        "surrogate": Surrogate(SimpleCustomGP),
                    },
                ),
            ]
        )

        ax_client = AxClient(generation_strategy=gs)
        ax_client.create_experiment(
        name="Ax_Tune_Pythia",
                parameters = [
                            {
                                "name": "aLund",
                                "type": "range",
                                "bounds": [0.0, 2.0],
                            }, 
                            {
                                "name": "bLund",
                                "type": "range",
                                "bounds": [0.2, 2.0],
                            },

                            # {
                            #     "name": "aExtraSQuark",
                            #     "type": "range",
                            #     "bounds": [0.0, 2.0],
                            # }, 

                            # {
                            #     "name": "sigma",
                            #     "type": "range",
                            #     "bounds": [0.,1.],
                            # }, 
                            # {
                            #     "name": "enhancedFraction",
                            #     "type": "range",
                            #     "bounds": [0.,1.],
                            # }, 

                            {
                                "name": "ProbStoUD",
                                "type": "range",
                                "value_type": "float",
                                "bounds": [0.0,1.0],
                            }, 
                            {
                                "name": "probQQtoQ",
                                "type": "range",
                                "value_type": "float",
                                "bounds": [0.0,1.0],
                            }, 
                            # {
                            #     "name": "probSQtoQQ",
                            #     "type": "range",
                            #     "value_type": "float",
                            #     "bounds": [0,4.0],
                            # }, 
        
                            {
                                "name": "alphaSvalue",
                                "type": "range",
                                "value_type": "float",
                                "bounds": [0.06,0.25],
                            }, 
                            {
                                "name": "pTmin",
                                "type": "range",
                                "value_type": "float",
                                "bounds": [0.1,2.0],
                            }, 
                    
                        ],
        objectives = {"pythia_objective_func": ObjectiveProperties(minimize=True)},
        )
        SUGGEST_MONASH=False
        if SUGGEST_MONASH:
            suggest_param, suggest_ind = ax_client.attach_trial(
                parameters=MONASH_DICT
            )
            ax_client.complete_trial(trial_index=suggest_ind, raw_data=pythia_objective_func(
                    aLund=suggest_param["aLund"], 
                    bLund=suggest_param["bLund"],
                    # aExtraSQuark=suggest_param["aExtraSQuark"],
                    # sigma=suggest_param["sigma"],
                    # enhancedFraction=suggest_param["enhancedFraction"],
                    ProbStoUD=suggest_param["ProbStoUD"],
                    probQQtoQ=suggest_param["probQQtoQ"],
                    # probSQtoQQ=suggest_param["probSQtoQQ"],
                    alphaSvalue=suggest_param["alphaSvalue"],
                    pTmin=suggest_param["pTmin"]
                    ))

        N_TOTAL_ITERATIONS = N_TRAIN_POINTS + n_bo_iterations
        for i in range(N_TOTAL_ITERATIONS):
            parameterization, trial_index = ax_client.get_next_trial()
            print(parameterization)
            ax_client.complete_trial(trial_index=trial_index, raw_data=pythia_objective_func(
            aLund=parameterization["aLund"], 
            bLund=parameterization["bLund"],
            # aExtraSQuark=parameterization["aExtraSQuark"],
            # sigma=parameterization["sigma"],
            # enhancedFraction=parameterization["enhancedFraction"],
            ProbStoUD=parameterization["ProbStoUD"],
            probQQtoQ=parameterization["probQQtoQ"],
            # probSQtoQQ=parameterization["probSQtoQQ"],
            alphaSvalue=parameterization["alphaSvalue"],
            pTmin=parameterization["pTmin"]
        ))
            
        
        BO_dir_string = f'BOTorch_pythia_N_TRAIN_POINTS_{N_TRAIN_POINTS}_N_BO_ITERATIONS_{n_bo_iterations}_N_PYTHIA_EVENTS_{NUM_PYTHIA_EVENTS}'
        BO_dir = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'output', BO_dir_string)
        if not os.path.exists(BO_dir):
            os.makedirs(BO_dir)
        # save best parameters
        best_parameters, values = ax_client.get_best_parameters()
        best_parameters_df = pd.DataFrame(best_parameters, index=[0])
        best_parameters_df.to_csv(os.path.join(BO_dir, 'best_parameters.csv'))

        model = ax_client.generation_strategy.model
        history_df = ax_client.get_trials_data_frame()
        history_df.to_csv(os.path.join(BO_dir, 'history.csv'))


        fig, ax = plt.subplots(3, 2, figsize=(20,15))
        ax = ax.ravel()
        params = list(MONASH_DICT.keys())
        for i, axi in enumerate(ax):
            
            BoTorch_plot_model_f_vs_param(model=model, param=params[i], ax=axi, history_df=history_df, reference_dict=MONASH_DICT)
            
            plt.tight_layout()
            plt.savefig(os.path.join(BO_dir, 'BOTorch_plot.pdf'))

if __name__ == "__main__":
    main()

