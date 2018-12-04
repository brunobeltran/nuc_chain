
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
from MultiPoint import propagator
#HETEROGENOUS CHAINS +/1 BP NOISE: NEX
#%%time
if __name__ == '__main__':
    #this is the same-ish range that Quinn used
    Klin = np.linspace(0, 10**5, 20000)
    Klog = np.logspace(-3, 5, 10000)
    Kvals = np.unique(np.concatenate((Klin, Klog)))
    #convert to little k -- units of inverse bp (this results in kmax = 332)
    kvals = Kvals / (2*wlc.default_lp)
    rvals = np.linspace(0.0, 1.0, 1000)
    # Kprops_bareWLC = wlc.tabulate_bareWLC_propagators(Kvals)
    # print('Tabulated K propagators for bare WLC!')
    unwrap = 0
    pool_size = 31
    chains = np.arange(21, 31)
    for i in chains:
        filepath = f'csvs/Bprops/0unwraps/heterogenous/links31to52/25nucs_chain{i}'
        #sample uniformly from one full period [31, 51]bp inclusive
        links31to52 = np.random.randint(31, 52, 25)
        ldna = ncg.genomic_length_from_links_unwraps(links31to52, unwraps=0)
        #while (ldna[-1] < 4675):
        #    links31to52 = np.concatenate((links31to52,
        #                                  np.random.randint(31,52,1)))
        #    ldna = ncg.genomic_length_from_links_unwraps(links31to52,
        #                                                   unwraps=0)
        with Pool(pool_size) as pool:
            pool.map(partial(wlc.Bprop_k_given_L, links=links31to52,
                            unwraps=unwrap, filepath=filepath), kvals)
        wlc.combine_Bprop_files(filepath, links31to52, unwrap)
