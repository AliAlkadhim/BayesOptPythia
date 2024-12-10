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

from BayesOpt_utils import *
from objective_funcs import *
from acquisition_funcs import *
from models import * 
from plotting import *
from yoda2numpy_BayesOpt import Yoda2Numpy
from pythia_SBI_utils import *
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

# NUM_PYTHIA_EVENTS=250
NUM_PYTHIA_EVENTS=250000

object_func = pythia_objective_func
def make_pythia_monash_card(MONASH_DICT):
    
    BO_Cards_dir = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'BO_Cards')
    if not os.path.exists(BO_Cards_dir):
        os.makedirs(BO_Cards_dir)
    
    filename = f"ALEPH_1996_S3486095_MONASH_card.cmnd"
    file_path = os.path.join(BO_Cards_dir, filename)

    with open(file_path,'w') as f:
        first_block=f"""Main:numberOfEvents = {NUM_PYTHIA_EVENTS}          ! number of events to generate
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

        f.write("StringZ:aLund = {}\n\n".format(MONASH_DICT["aLund"]))
        f.write("StringZ:bLund = {}\n\n".format(MONASH_DICT["bLund"]))
        # f.write("StringZ:rFactC = {}\n\n".format(best_parameters["rFactC"]))
        # f.write("StringZ:rFactB = {}\n\n".format(best_parameters["rFactB"]))
        # f.write("StringZ:aExtraSQuark = {}\n\n".format(MONASH_DICT["aExtraSQuark"]))
        # f.write("StringZ:aExtraDiquark = {}\n\n".format(best_parameters["aExtraDiquark"]))
        # f.write("StringPT:sigma = {}\n\n".format(MONASH_DICT["sigma"]))
        # f.write("StringPT:enhancedFraction = {}\n\n".format(MONASH_DICT["enhancedFraction"]))
        # f.write("StringPT:enhancedWidth = {}\n\n".format(best_parameters["enhancedWidth"]))
        f.write("StringFlav:ProbStoUD = {}\n\n".format(MONASH_DICT["ProbStoUD"]))
        f.write("StringFlav:probQQtoQ = {}\n\n".format(MONASH_DICT["probQQtoQ"]))
        # f.write("StringFlav:probSQtoQQ = {}\n\n".format(MONASH_DICT["probSQtoQQ"]))
        # f.write("StringFlav:ProbQQ1toQQ0 = {}\n\n".format(best_parameters["ProbQQ1toQQ0"]))
        f.write("TimeShower:alphaSvalue = {}\n\n".format(MONASH_DICT["alphaSvalue"]))
        f.write("TimeShower:pTmin = {}\n\n".format(MONASH_DICT["pTmin"]))
        
def run_monash_card(MONASH_DICT):
    
    # step 1: write .cmnd file 
    make_pythia_monash_card(MONASH_DICT)
    #step 2 run main42 and rivet
    main42_path = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'src', 'main42')
    BO_card_path = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'BO_Cards', 'ALEPH_1996_S3486095_MONASH_card.cmnd')
    temp_path = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'temp')
    monash_ALEPH_YODAS_path = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'data', 'ALEPH_YODAS_BayesOpt')

    os.system(f"""{main42_path} {BO_card_path} {temp_path}/ALEPH_1996_S3486095_MONASH_card.fifo
    rivet -o ALEPH_1996_S3486095_hist_monash_0.yoda -a ALEPH_1996_S3486095 {temp_path}/ALEPH_1996_S3486095_MONASH_card.fifo

    rm {temp_path}/ALEPH_1996_S3486095_MONASH_card.fifo
    mv ALEPH_1996_S3486095_hist_monash_0.yoda {monash_ALEPH_YODAS_path}""")
    



def get_monash_data():
    # tracemalloc.start()
    yoda2numpy = Yoda2Numpy()
    YODA_BASE = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'data', 'ALEPH_YODAS_BayesOpt')
    files = list(glob(f'{YODA_BASE}/*monash*.yoda'))
    print(files)
    M = len(files)
    # M = 
    generated_indices = []
    for file in files[:M]:
        index = file.split('_')[-1].split('.')[0]
        generated_indices.append(int(index))    
    generated_indices.sort()
    print(generated_indices)
    # # --- SIM
    print(f'looping over {M:d} sim yoda files...\n')
    # dfsims = []
    dfsims = {}
    for ii in tqdm(generated_indices):    
        # index here should match the index of the file
        # dfsims.append( yoda2numpy.todf( yoda2numpy('sim', index=ii) ) )
        dfsims[ii]= yoda2numpy.todf( yoda2numpy('mon', index=ii) ) 
        # current, peak = tracemalloc.get_traced_memory() 
        # print(current/(1024*1024), 'MB')
    # tracemalloc.stop()

    # # --- NEW
    # print(f'looping over {M:d} new yoda files...\n')
    # # dfnews = []
    # # for ii in tqdm(range(M)):
    # #     dfnews.append( yoda2numpy.todf( yoda2numpy('new', index=ii) ) )

    print()
    # key = '/ALEPH_1996_S3486095/d01-x01-y01'
    # dfsim = dfsims[0][key]
    
    dfdata = yoda2numpy.todf( yoda2numpy('dat') )
    
    return dfdata, dfsims, generated_indices

def get_EI_valid_data():
    # tracemalloc.start()
    yoda2numpy = Yoda2Numpy()
    YODA_BASE = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'data', 'ALEPH_YODAS_BayesOpt')
    files = list(glob(f'{YODA_BASE}/*EI_valid*.yoda'))
    print(files)
    M = len(files)
    # M = 
    generated_indices = []
    for file in files[:M]:
        index = file.split('_')[-1].split('.')[0]
        generated_indices.append(int(index))    
    generated_indices.sort()
    print(generated_indices)
    # # --- SIM
    print(f'looping over {M:d} sim yoda files...\n')
    # dfsims = []
    dfsims = {}
    for ii in tqdm(generated_indices):    
        # index here should match the index of the file
        # dfsims.append( yoda2numpy.todf( yoda2numpy('sim', index=ii) ) )
        dfsims[ii]= yoda2numpy.todf( yoda2numpy('val', index=ii) ) 
        # current, peak = tracemalloc.get_traced_memory() 
        # print(current/(1024*1024), 'MB')
    # tracemalloc.stop()

    # # --- NEW
    # print(f'looping over {M:d} new yoda files...\n')
    # # dfnews = []
    # # for ii in tqdm(range(M)):
    # #     dfnews.append( yoda2numpy.todf( yoda2numpy('new', index=ii) ) )

    print()
    # key = '/ALEPH_1996_S3486095/d01-x01-y01'
    # dfsim = dfsims[0][key]
    
    dfdata = yoda2numpy.todf( yoda2numpy('dat') )
    
    return dfdata, dfsims, generated_indices

def get_qEI_valid_data():
    # tracemalloc.start()
    yoda2numpy = Yoda2Numpy()
    YODA_BASE = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'data', 'ALEPH_YODAS_BayesOpt')
    files = list(glob(f'{YODA_BASE}/*qEI_valid*.yoda'))
    print(files)
    M = len(files)
    # M = 
    generated_indices = []
    for file in files[:M]:
        index = file.split('_')[-1].split('.')[0]
        generated_indices.append(int(index))    
    generated_indices.sort()
    print(generated_indices)
    # # --- SIM
    print(f'looping over {M:d} sim yoda files...\n')
    # dfsims = []
    dfsims = {}
    for ii in tqdm(generated_indices):    
        # index here should match the index of the file
        # dfsims.append( yoda2numpy.todf( yoda2numpy('sim', index=ii) ) )
        dfsims[ii]= yoda2numpy.todf( yoda2numpy('qEI', index=ii) ) 
        # current, peak = tracemalloc.get_traced_memory() 
        # print(current/(1024*1024), 'MB')
    # tracemalloc.stop()

    # # --- NEW
    # print(f'looping over {M:d} new yoda files...\n')
    # # dfnews = []
    # # for ii in tqdm(range(M)):
    # #     dfnews.append( yoda2numpy.todf( yoda2numpy('new', index=ii) ) )

    print()
    # key = '/ALEPH_1996_S3486095/d01-x01-y01'
    # dfsim = dfsims[0][key]
    
    dfdata = yoda2numpy.todf( yoda2numpy('dat') )
    
    return dfdata, dfsims, generated_indices


def make_pythia_valid_card(best_params_df, BOTorch=False):
    
    BO_Cards_dir = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'BO_Cards')
    if not os.path.exists(BO_Cards_dir):
        os.makedirs(BO_Cards_dir)
    if BOTorch:
        filename = f"ALEPH_1996_S3486095_qEI_valid_card.cmnd"
    else:
        filename = f"ALEPH_1996_S3486095_EI_valid_card.cmnd"
    file_path = os.path.join(BO_Cards_dir, filename)

    with open(file_path,'w') as f:
        first_block=f"""Main:numberOfEvents = {NUM_PYTHIA_EVENTS}          ! number of events to generate
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

        f.write("StringZ:aLund = {}\n\n".format(float(best_params_df["aLund"])))
        f.write("StringZ:bLund = {}\n\n".format(float(best_params_df["bLund"])))
        # f.write("StringZ:rFactC = {}\n\n".format(best_parameters["rFactC"]))
        # f.write("StringZ:rFactB = {}\n\n".format(best_parameters["rFactB"]))
        # f.write("StringZ:aExtraSQuark = {}\n\n".format(MONASH_DICT["aExtraSQuark"]))
        # f.write("StringZ:aExtraDiquark = {}\n\n".format(best_parameters["aExtraDiquark"]))
        # f.write("StringPT:sigma = {}\n\n".format(MONASH_DICT["sigma"]))
        # f.write("StringPT:enhancedFraction = {}\n\n".format(MONASH_DICT["enhancedFraction"]))
        # f.write("StringPT:enhancedWidth = {}\n\n".format(best_parameters["enhancedWidth"]))
        f.write("StringFlav:ProbStoUD = {}\n\n".format(float(best_params_df["ProbStoUD"] )))
        f.write("StringFlav:probQQtoQ = {}\n\n".format(float(best_params_df["probQQtoQ"])))
        # f.write("StringFlav:probSQtoQQ = {}\n\n".format(MONASH_DICT["probSQtoQQ"]))
        # f.write("StringFlav:ProbQQ1toQQ0 = {}\n\n".format(best_parameters["ProbQQ1toQQ0"]))
        f.write("TimeShower:alphaSvalue = {}\n\n".format(float(best_params_df["alphaSvalue"])))
        f.write("TimeShower:pTmin = {}\n\n".format(float(best_params_df["pTmin"])))


def run_EI_valid_card(best_params_df):
    
    # step 1: write .cmnd file 
    make_pythia_valid_card(best_params_df, BOTorch=False)
    #step 2 run main42 and rivet
    main42_path = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'src', 'main42')
    BO_card_path = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'BO_Cards', 'ALEPH_1996_S3486095_EI_valid_card.cmnd')
    temp_path = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'temp')
    EI_valid_ALEPH_YODAS_path = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'data', 'ALEPH_YODAS_BayesOpt')

    os.system(f"""{main42_path} {BO_card_path} {temp_path}/ALEPH_1996_S3486095_EI_valid_card.fifo
    rivet -o ALEPH_1996_S3486095_hist_EI_valid_0.yoda -a ALEPH_1996_S3486095 {temp_path}/ALEPH_1996_S3486095_EI_valid_card.fifo

    rm {temp_path}/ALEPH_1996_S3486095_EI_valid_card.fifo
    mv ALEPH_1996_S3486095_hist_EI_valid_0.yoda {EI_valid_ALEPH_YODAS_path}""")


def run_qEI_valid_card(best_params_df):
    
    # step 1: write .cmnd file 
    make_pythia_valid_card(best_params_df, BOTorch=True)
    #step 2 run main42 and rivet
    main42_path = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'src', 'main42')
    BO_card_path = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'BO_Cards', 'ALEPH_1996_S3486095_qEI_valid_card.cmnd')
    temp_path = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'temp')
    EI_valid_ALEPH_YODAS_path = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'data', 'ALEPH_YODAS_BayesOpt')

    os.system(f"""{main42_path} {BO_card_path} {temp_path}/ALEPH_1996_S3486095_qEI_valid_card.fifo
    rivet -o ALEPH_1996_S3486095_hist_qEI_valid_0.yoda -a ALEPH_1996_S3486095 {temp_path}/ALEPH_1996_S3486095_qEI_valid_card.fifo

    rm {temp_path}/ALEPH_1996_S3486095_qEI_valid_card.fifo
    mv ALEPH_1996_S3486095_hist_qEI_valid_0.yoda {EI_valid_ALEPH_YODAS_path}""")

if __name__ == "__main__":
    #################### RUN MONASH AND VALIDATION
    





    #################### ANALYSE MONASH AND VALIDATION
    # run_monash_card(MONASH_DICT)

    dfdata, dfsims_monash, generated_indices = get_monash_data()
    data_keys, mc_keys = get_hist_names(dfdata)
    filtered_data_keys, filtered_mc_keys = filter_keys(dfdata, dfsims_monash, data_keys, mc_keys)
    reduced_data_keys, reduced_mc_keys = reduce_filtered_keys(filtered_data_keys, filtered_mc_keys)

    hists_monash = make_hists(dfdata, dfsims_monash[0], reduced_data_keys, reduced_mc_keys)


    ############ EI VALIDATION
    configs_dict = {
    'N_BO_ITERATIONS': 80,
    'N_TRAIN_POINTS': 25,
    'N_PARAMS': 6,
    'N_OPTIMIZE_ACQ_ITER': 50,
    'N_RESTARTS': 25,
    'KERNEL': 'Matern',
    'OPTIMIZE_ACQ_METHOD':  'Adam_restarts_clip_bounds',
    'NUM_PYTHIA_EVENTS': 250000,
    }
    _, config_string = configs_df(configs_dict)
    print(config_string)
    dir_name = directory_name(object_func, configs_dict)
    # print(f'dir_name={dir_name}')
    path_name = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'output', dir_name)
    best_params_df = load_best_params_df(path_name)
    print(best_params_df.head())
    # print(best_params_df["aLund"])
    run_EI_valid_card(best_params_df)

    dfdata, dfsims_EI_valid, generated_indices = get_EI_valid_data()
    hists_EI_valid = make_hists(dfdata, dfsims_EI_valid[0], reduced_data_keys, reduced_mc_keys)

    EI_valid_plot_path = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'output','post_processing', f"ALEPH_1996_S3486095_EI_valid_plot_N_TRAIN_POINTS_{configs_dict['N_TRAIN_POINTS']}_N_BO_ITERATIONS_{configs_dict['N_BO_ITERATIONS']}_reduced_objective_func_all_hists.pdf")

    plot_dist_with_monash(hist_names=reduced_data_keys, monash_hists=hists_monash, valid_hists=hists_EI_valid, filename=EI_valid_plot_path)


    ################### ANALYSE BOTORCH
    # N_TRAIN_POINTS = 25
    # n_bo_iterations = 90

    # N_TRAIN_POINTS = 25
    # n_bo_iterations = 120

    # BO_dir_string = f'BOTorch_pythia_N_TRAIN_POINTS_{N_TRAIN_POINTS}_N_BO_ITERATIONS_{n_bo_iterations}_N_PYTHIA_EVENTS_{NUM_PYTHIA_EVENTS}'
    # BO_dir = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'output', BO_dir_string)
    # best_params_df = load_best_params_df(BO_dir, BOTorch=True)
    # print(best_params_df.head())
    # run_qEI_valid_card(best_params_df)

    # dfdata, dfsims_qEI_valid, generated_indices = get_qEI_valid_data()
    # hists_qEI_valid = make_hists(dfdata, dfsims_qEI_valid[0], reduced_data_keys, reduced_mc_keys)

    # qEI_valid_plot_path = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'output','post_processing', f'ALEPH_1996_S3486095_qEI_valid_plot_N_TRAIN_POINTS_{N_TRAIN_POINTS}_N_BO_ITERATIONS_{n_bo_iterations}_reduced_objective_func_all_hists.pdf')

    # plot_dist_with_monash(hist_names=reduced_data_keys, monash_hists=hists_monash, valid_hists=hists_qEI_valid, filename=qEI_valid_plot_path, BOTorch=True)

