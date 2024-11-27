import os, sys, re
import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import importlib
# from joblib import  Memory
import tracemalloc

#`source /cvmfs/sft.cern.ch/lcg/views/LCG_102/x86_64-centos7-gcc11-opt/setup.sh`
from glob import glob
from tqdm import tqdm
from yoda2numpy_BayesOpt import Yoda2Numpy

# update fonts
FONTSIZE = 14
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}
mp.rc('font', **font)

# set usetex = False if LaTex is not 
# available on your system or if the 
# rendering is too slow
mp.rc('text', usetex=True)
BAYESOPT_BASE=os.environ['BAYESOPT_BASE']


# memory = Memory('/afs/cern.ch/work/a/aalkadhi/private/TUNES/Pythia8_SBI_Tune/cluster')
# memory = Memory('/home/ali/Downloads')


# @memory.cache
def get_data():
    # tracemalloc.start()
    yoda2numpy = Yoda2Numpy()
    YODA_BASE = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'data', 'ALEPH_YODAS_BayesOpt')
    files = list(glob(f'{YODA_BASE}/*.yoda'))
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
        dfsims[ii]= yoda2numpy.todf( yoda2numpy('sim', index=ii) ) 
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


def get_hist_names(dfdata):
    data_keys = list(dfdata.keys())
    mc_keys = [data_key[4:] for data_key in data_keys]
    
    return data_keys, mc_keys


def filter_keys(dfdata, dfsims, data_keys, mc_keys):
    new_data_keys = []
    new_mc_keys = []
    n_dfsims = len(dfsims)
    for data_key, mc_key in zip(data_keys, mc_keys):
        dfdat = dfdata[data_key]
        yval = dfdat['yval']
        yerr = dfdat['yerr-']
        
        if len(yval) < 2: continue
        # for i in range(n_dfsims):
        if not (mc_key in list(dfsims.values())[0]): continue
        # if not all(mc_key in sim for sim in list(dfsims.values())): continue
        # if not all(mc_key in list(dfsims.values())[i] for i in range(n_dfsims)): continue
        for key, val in dfsims.items():
            if not mc_key in list(val):
                print('keys dont match in file index', key)
        
        new_data_keys.append(data_key)
        new_mc_keys.append(mc_key)
        
    print(' new data keys', new_data_keys) 
    print(' new mc keys', new_mc_keys) 
    return new_data_keys, new_mc_keys

# def filter_keys(dfdata, dfsims, data_keys, mc_keys):
#     new_data_keys = []
#     new_mc_keys = []
#     n_dfsims = len(dfsims)
#     for data_key, mc_key in zip(data_keys, mc_keys):
#         dfdat = dfdata[data_key]
#         yval = dfdat['yval']
#         yerr = dfdat['yerr-']
        
#         if len(yval) >= 2:
#             # for i in range(n_dfsims):
#             if (mc_key in dfsims[0]):
#                 new_data_keys.append(data_key)
#                 new_mc_keys.append(mc_key)
        
        
#     new_data_keys = list(set(new_data_keys))
#     new_mc_keys = list(set(new_mc_keys))
#     print(' new data keys', new_data_keys) 
#     print(' new mc keys', new_mc_keys) 
    
    
#     print('len new data keys', len(new_data_keys)) 
#     print('len new mc keys', len(new_mc_keys)) 
#     # overlap = new_data_keys.intersection(new_mc_keys)
#     # print('len overlap keys', len(overlap)) 
#     return new_data_keys, new_mc_keys


def test_statistic(data_keys, mc_keys, dfdata, dfpred, which = 0):
    Y=0.0
    nbins=0
    # if which=0, we are using dfdata=dfdata, 
    #if which =1 we are using dfdata=dfbest or dfdata=dfsim
    
    for data_key, mc_key in zip(data_keys, mc_keys):
        if which == 0:
            df = dfdata[data_key]
            data = df['yval']
            data_err = df['yerr-']
        else:
            df = dfdata[mc_key]
            data = df['sumw']
            data_err = np.sqrt(df['sumw2'])
            
        ndat = len(df)
        pred_df = dfpred[mc_key]
        npred = len(pred_df)
        
        assert ndat == npred
        
        pred = pred_df['sumw']
        pred_err2 = pred_df['sumw2']
        
        stdv = np.sqrt(data_err**2 + pred_err2)
        stdv = np.where(stdv < 1e-3, 1, stdv)
        
        # weight pred by 1/Neff
        # N_eff = pred**2 / pred_err2
        # pred = 1/N_eff * pred
        
        X = (((data-pred)/stdv)**2).sum()
        Y += X
        nbins += ndat
        
    # if nbins == 0:
    #     raise ZeroDivisionError("nbins = 0 in test statistic")
        
    # nbins = np.where(nbins < 1e-3, 1, nbins)
    # return np.sqrt(Y/nbins)
    return Y


def make_hists(dfdata, dfbest, filtered_data_keys, filtered_mc_keys):
    hists = []
    for data_key, mc_key in zip(filtered_data_keys, filtered_mc_keys):
        data=dfdata[data_key]
        data_val = data['yval'].values
        # data_err = data['yerr-']
        
        pred = dfbest[mc_key]
        pred_val = pred['sumw'].values
        # pred_err = np.sqrt(pred_val['sumw2'])
        edges = dfbest[mc_key]['xlow'].values
        
        hist_name = data_key.split('/')[-1]
        hists.append((hist_name, edges, data_val, pred_val))
    return hists

def plt_sim_data_hist(ax, hist):
    name, edges, data_val, pred_val = hist
    
    xmin = 0
    xmax = edges.max()
    ax.set_ylim(xmin, xmax)
    
    ymin = 0
    ymax = 1.25 * data_val.max()
    ax.set_ylim(ymin, ymax)

    ax.text(xmin + 0.7*(xmax-xmin), ymin + 0.2*(ymax-ymin), name)
    
    ax.step(y=data_val, x=edges, label='data')
    ax.step(y=pred_val, x=edges, label='pred')
    ax.legend()
            
def plot_dist(hist_names, hists, filename='fig_bestfit_dist_all.png'):
    
        
    nhists= len(hist_names)
    ncols = 3
    nrows = nhists // ncols
    nhists= nrows * ncols
    
    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 20), edgecolor='k')
    ax = ax.ravel()
    for hist_ind, hist in enumerate(hists[:nhists]):
        plt_sim_data_hist(ax[hist_ind], hist)
    
    plt.tight_layout()
    plt.savefig(filename)   
    
    plt.show()
        
    
    
    

def plot_cdf(df, a=None, b=None,
             xbins=10, xmin=0.5, xmax=1.5, 
             ybins=10, ymin=0.5, ymax=1.5, 
             filename='fig_cdf_via_hist.png',
             fgsize=(5, 5), 
             ftsize=18):
    
    # approximate cdf via histogramming
    xrange = (xmin, xmax)
    yrange = (ymin, ymax)
        
    # weighted histogram   (count the number of ones per bin)
    hw, xe, ye = np.histogram2d(df['StringZ:aLund'], df['StringZ:bLund'], 
                                bins=(xbins, ybins), 
                                range=(xrange, yrange), 
                                weights=df.Z0)

    # unweighted histogram (count number of ones and zeros per bin)
    hu, _, _ = np.histogram2d(df['StringZ:aLund'], df['StringZ:bLund'], 
                              bins=(xbins, ybins), 
                              range=(xrange, yrange)) 
    P =  hw / (hu + 1.e-10)    
  
    # flatten arrays so that p, x, and y are 1d arrays
    # of the same length.    
    # get bin centers
    x   = (xe[1:] + xe[:-1])/2
    y   = (ye[1:] + ye[:-1])/2
    X,Y = np.meshgrid(x, y)
    x   = X.flatten()
    y   = Y.flatten()
    
    # WARNING: must transpose P so that X, Y, and P have the
    # same shape
    P   = P.T
    p   = P.flatten()
                
    # Now make plots
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fgsize)
    
    ax.set_xlim(xmin, xmax)
    #ax.set_xticks(np.linspace(xmin, xmax, 6))
    ax.set_xlabel(r'$a$', fontsize=ftsize)
    
    ax.set_ylim(ymin, ymax)
    #ax.set_yticks(np.linspace(xmin, xmax, 9))
    ax.set_ylabel(r'$b$',  fontsize=ftsize)
    
    mylevels = np.array([0.68, 0.8, 0.9])

    colormap = 'rainbow'
    
    cs = ax.contour(X, Y, P, 
                    extent=(xmin, xmax, ymin, ymax),
                    levels=mylevels,
                    linewidths=2,
                    linestyles='dashed',
                    cmap=colormap)

    ax.clabel(cs, cs.levels, 
              inline=True, 
              fontsize=18, fmt='%4.2f', 
              colors='black')

    if a != None:
        if b != None:
            print(f'a(best): {a_best:10.3f} b(best): {b_best:10.3f}')
            ax.plot([a_best], [b_best], 
                    markerfacecolor='red',
                    markersize=20, 
                    marker='.')
            
    ax.grid()

    plt.tight_layout()
    # plt.savefig(filename)    
    plt.show()
    
def find_key_by_value(my_dict, search_value):
    for key, value in my_dict.items():
        if value == search_value:
            return int(key)
    return None  # If the value is not found, return None

    
if __name__ == '__main__':
    dfdata, dfsims, generated_indices = get_data()
    print(generated_indices)
    # print(dfsims)
    print('DATA DATAFRAME')
    print(dfdata['/REF/ALEPH_1996_S3486095/d01-x01-y01'].head())
    print('FIRST SIM DATAFRAME')
    print(dfsims[generated_indices[0]]['/ALEPH_1996_S3486095/d01-x01-y01'].head())
    
    data_keys, mc_keys = get_hist_names(dfdata)
    filtered_data_keys, filtered_mc_keys = filter_keys(dfdata, dfsims, data_keys, mc_keys)
    
    
    # X0 = []
    # indices_giving_error = []
    # X0 = {}
    # for ii, gen_ind in enumerate(generated_indices):
    #     # X0.append(test_statistic(filtered_data_keys,filtered_mc_keys, dfdata, dfsims[gen_ind], which = 0))
    #     # try:
    #     #     X0.append(test_statistic(filtered_data_keys,filtered_mc_keys, dfdata, dfsims[ii], which = 0))
    #     try:
    #         X0[gen_ind] = test_statistic(filtered_data_keys,filtered_mc_keys, dfdata, dfsims[gen_ind], which = 0)
    #     except Exception:
    #         print('test statistic error in file index: ', gen_ind)
    #     # except ZeroDivisionError:
    #     #     print(f"gen_index {gen_ind} somehow gives nbins=0")
    #     #     indices_giving_error.append(gen_ind)
    #     #     continue
    # print('indices_giving error: ', indices_giving_error.sort())
    # X0_arr = np.array(list(X0.values()))
    # print('X0 array:', X0_arr)
    # # plt.hist(X0_arr); plt.show()
    # df_all_params = pd.read_csv('LHS_DF_20000.csv')
    # print(f'generated indices: {generated_indices}')
    # df_all_params=df_all_params.iloc[generated_indices].reset_index()
    # print(len(df_all_params))
    # print(df_all_params)
    
    # K_best = np.argmin(X0_arr) # file with minimum value of test statistic
    # lowest_X0 = X0_arr[K_best]
    # index_of_lowest_X0 = find_key_by_value(X0, lowest_X0)
    
    # print(f'best index of X0 (not index of file): {K_best}')
    
    # print(f'file index of lowest X0: {index_of_lowest_X0}')
    
    # dfbest = dfsims[index_of_lowest_X0]
    
    # assert len(df_all_params) == len(X0_arr)
    # print(df_all_params.iloc[K_best])
    
    # a_best = df_all_params['StringZ:aLund'].iloc[K_best]
    # b_best = df_all_params['StringZ:bLund'].iloc[K_best]
    
    # print(f'a best= {a_best}, b best = {b_best}')
    
    # hists = make_hists(dfdata, dfbest, filtered_data_keys, filtered_mc_keys)
    # plot_dist(filtered_data_keys, hists, filename='fig_bestfit_dist.png')
    
    
    