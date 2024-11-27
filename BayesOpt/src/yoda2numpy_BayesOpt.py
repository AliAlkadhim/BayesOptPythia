# ----------------------------------------------------------------------------
# Created Jan. 30 2024 HBP
# ----------------------------------------------------------------------------
import os, sys, re
import numpy as np
import pandas as pd
BAYESOPT_BASE=os.environ['BAYESOPT_BASE']

#from numba import njit
# ----------------------------------------------------------------------------

def decode_yoda(hists, findlabel, findbegin, findnumber):
    labels = None
    hmap = {}
    
    for ii, hist in enumerate(hists):

        # make sure we have at least 6 columns of numbers
        label = findlabel.findall(hist)
        if len(label) == 0: continue
        delim = label[0] # we'll split the string at this delimeter
        label = delim.split()
        nlabel= len(label)
        if nlabel < 6: continue

        labels = label # cache labels
       
        # split string into two pieces
        header, values = hist.split(delim)

        # find name of histogram (on BEGIN line)
        name = findbegin.findall(header)[0].split()[-1]
        
        # convert substrings to floats, store data in an array and reshape array 
        # to shape (number-of-bins, number-of-columns)
        d = np.array([float(x) for x in findnumber.findall(values)]).reshape(-1, nlabel)

        # check whether to convert to differential cross sections
        if labels[0] == 'xlow':
            # convert to differential cross sections by 
            # dividing by bin widths
            dx = d.T[1]-d.T[0]
            dx2= dx*dx
            d.T[2] /= dx
            d.T[3] /= dx2
            d.T[4] /= dx
            d.T[5] /= dx2
            
        # store data in a map
        if name in hmap:
            raise ValueError('duplicate key: '+name)
        else:
            hmap[name] = d
            
    return labels, hmap
    
class Yoda2Numpy:
    
    def __init__(self):
        # find BEGIN..END blocks
        self.findhist  = re.compile(r'BEGIN.*?END', re.DOTALL)
        
        self.findnumber= re.compile(r'[0-9][.][0-9]+[e][-+][0-9]+', re.DOTALL)

        # find column labels of histogram data
        self.findlabel = re.compile(r'(?<=[#] )xlow.+|(?<=[#] )xval.+')

        # find name of histogram
        self.findbegin  = re.compile(r'(?<=BEGIN ).+')

    def pathname(self, hist_type, 
                 index=0, 
                 filename='ALEPH_1996_S3486095'):
        
        YODA_BASE = os.path.join(BAYESOPT_BASE, 'BayesOpt', 'data', 'ALEPH_YODAS_BayesOpt')
        htype = hist_type[:3]
        if htype == 'sim':
            yoda_dir = ''
            postfix  = f'_hist_{index:d}'
            return f'{YODA_BASE:s}/{filename:s}{postfix:s}.yoda'
        
        if htype == 'val':
            yoda_dir = ''
            postfix  = f'_hist_valid_{index:d}'
            return f'{YODA_BASE:s}/{filename:s}{postfix:s}.yoda'
        
        elif htype == 'dat':
            yoda_dir = 'data'
            postfix  = ''
            return f'{YODA_BASE:s}/{yoda_dir:s}/{filename:s}{postfix:s}.yoda'
            

        # elif htype == 'new':
        #     yoda_dir = 'newseeds'
        #     postfix  = f'_card_newseed_{index:d}'
        else:
            raise ValueError(f'hist_type {hist_type:s} unknown')
            return None
            
        
        
    def __call__(self, hist_type, 
                 index=0, 
                 fname='ALEPH_1996_S3486095', 
                 first=1, 
                 last=44):
        filename = self.pathname(hist_type, index, fname)
        print("using filename", filename)

        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        findhist  = self.findhist
        findnumber= self.findnumber
        findlabel = self.findlabel
        findbegin = self.findbegin

        # read yoda file into a single string
        record= open(filename).read()
        hists = findhist.findall(record)
        
        label, hmap = decode_yoda(hists, findlabel, findbegin, findnumber)

        return label, hmap

    def todf(self, hists):
        labels, hmap = hists
        dfmap = {}
        keys  = list(hmap.keys())
        for key in keys:
            h = hmap[key].T
            df_at_key =pd.DataFrame({label: value for label, value in zip(labels, h)})
            df_at_key= df_at_key.astype('float32')
            dfmap[key] = df_at_key
        return dfmap
        