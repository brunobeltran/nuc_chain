
import pickle
import functools
from functools import partial
import itertools
import imp
import multiprocessing
from multiprocessing import Pool
import re
import os
from pathlib import Path
import sys
sys.path.insert(0, '/home/users/dkannan/git-remotes/deepti')

import numpy as np
import scipy
import pandas as pd

import bruno_util
from bruno_util import pandas_util
from bruno_util import pickle_variables
from nuc_chain import geometry as ncg
from nuc_chain import data as ncd
from nuc_chain import rotations as ncr
from nuc_chain import wignerD as wd
from nuc_chain import fluctuations as wlc
from nuc_chain import linkers as ncl
from MultiPoint import propagator
#HETEROGENOUS CHAINS +/1 BP NOISE: NEXdd
#%%time
if __name__ == '__main__':
    #this is the same-ish range that Quinn used
    Klin = np.linspace(0, 10**5, 20000)
    Klog = np.logspace(-3, 5, 10000)
    Kvals = np.unique(np.concatenate((Klin, Klog)))
    #convert to little k -- units of inverse bp (this results in kmax = 332)
    kvals = Kvals / (2*wlc.default_lp)
    unwrap = 0
    i = sys.argv[1]
    filepath = f'csvs/Bprops/0unwraps/heterogenous/exp_mu56/50nucs_chain{i}'
    links = ncl.independent_linker_lengths(mu=56, size=50)
    print(links)
    sys.stdout.flush()
    for k in kvals:
        wlc.Bprop_k_given_L(k, links=links, unwraps=unwrap, filepath=filepath)
    wlc.combine_Bprop_files(filepath, links, unwrap)
    print('DONE')
    sys.stdout.flush()
