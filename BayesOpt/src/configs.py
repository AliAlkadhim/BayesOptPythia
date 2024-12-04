
PARAM_DICT = {
        'StringZ:aLund' : [0, 2.0],
        'StringZ:bLund': [0.2, 2.0],
        # 'StringZ:rFactC':[0.0, 2.0],
        # 'StringZ:rFactB': [0., 2.0],
        # 'StringZ:aExtraSQuark':[0.,2.],
        # 'StringZ:aExtraDiquark':[0.,2.],
        # 'StringPT:sigma':[0.,1.],
        # 'StringPT:enhancedFraction':[0.,1.],
        # 'StringPT:enhancedWidth':[1.0,10.0],
        'StringFlav:ProbStoUD':[0,1.0],
        'StringFlav:probQQtoQ':[0,1.0],
        # 'StringFlav:probSQtoQQ':[0,1.0],
        # 'StringFlav:ProbQQ1toQQ0':[0,1.0],
        'TimeShower:alphaSvalue':[0.06,0.25],
        'TimeShower:pTmin':[0.1,2.0]
}

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

param_names = list(PARAM_DICT.keys())
num_params = len(param_names)
POINT2 = {key:0.5*value for key, value in MONASH_DICT.items()}
POINT3 = {key:1.5*value for key, value in MONASH_DICT.items()}

# NUM_PYTHIA_EVENTS=250000
NUM_PYTHIA_EVENTS=1000
#250000, 1000
KERNEL = 'Matern'
NUM_TRAIN_POINTS=25
N_TRAIN_EPOCHS=300
N_BO_ITERATIONS = 70
#N_BO_ITERATIONS = 2
N_OPTIMIZE_ACQ_ITER = 50 
N_RESTARTS = 25
# OPTIMIZE_ACQ_METHOD: ['GD', 'Adam_restarts_clip_bounds', 'Adam_no_clip_bounds', 'scipy']
OPTIMIZE_ACQ_METHOD = 'Adam_restarts_clip_bounds'
KAPPA=-1E-5


