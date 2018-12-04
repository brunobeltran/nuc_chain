"""Calculate Greens functions and stuff for the nucleosome positions that Bruno
created for his quals for yeast chromosome I."""

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
from nuc_chain.linkers import convert
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

    ### 37/38 random heterogenous linkers!
    # links37_38 = np.random.randint(37, 39, 2) #random linkers between 37 and 38 bp
    links_from_bruno =  convert.links_from_dyad_locations('csvs/dyad_locations_goo_sim_pre_qual_1.csv',
                                 unwraps=0)
    #select out 50 monomers near the telomere (10% from the end)
    links1142to1192 = links_from_bruno[1142:1192]
    with Pool(31) as pool:
        #returns a list of B000(k;N=1) for each of the kvals
        pool.map(partial(wlc.Bprop_k_given_L, links=links1142to1192,
                         unwraps=0), kvals)
    #bprops should now be a matrix where rows are k values and columns are N values (50 of them)
    #bprops37_38 = np.array(bprops37_38)
    #np.save(f'csvs/Bprops/0unwraps/heterogenous/B0_k_given_N_links37or38_2nucs_30000Ks.npy', bprops37_38, allow_pickle=False)
    #np.save(f'csvs/Bprops/0unwraps/heterogenous/linker_lengths_37or38_2nucs.npy', links37_38, allow_pickle=False)
    # qprop37_38 = wlc.bareWLC_gprop(kvals, links37_38, unwrap, props=Kprops_bareWLC)
    # integral37_38 = wlc.BRN_fourier_integrand_splines(kvals, links37_38, unwrap, Bprop=bprops37_38, rvals=rvals) #default: 1000 rvals
    # qintegral37_38 = wlc.BRN_fourier_integrand_splines(kvals, links37_38, unwrap, Bprop=qprop37_38, rvals=rvals) #default: 1000 rvals
    # #integral takes ~10 min to run, so prob worth saving
    # np.save(f'csvs/Bprops/{unwrap}unwraps/heterogenous/kinkedWLC_greens_links37or38_{len(rvals)}rvals_50nucs.npy', integral37_38, allow_pickle=False)
    # np.save(f'csvs/Bprops/{unwrap}unwraps/heterogenous/bareWLC_greens_links37or38_{len(rvals)}rvals_50nucs.npy', qintegral37_38, allow_pickle=False)
    # print(f'Saved G(R;L) for 37/38 link, {unwrap} unwrap!')

    # ### 45 or 46 random heterogenous linkers!
    # links45_46 = np.random.randint(45, 47, 50) #random linkers between 37 and 38 bp
    # with Pool(31) as pool:
    #     #returns a list of B000(k;N=1) for each of the kvals
    #     bprops45_46 = pool.map(partial(wlc.Bprop_k_given_L, links=links45_46, unwraps=0), kvals)
    # #bprops should now be a matrix where rows are k values and columns are N values (50 of them)
    # bprops45_46 = np.array(bprops45_46)
    # np.save(f'csvs/Bprops/0unwraps/heterogenous/B0_k_given_N_links45or46_50nucs_30000Ks.npy', bprops45_46, allow_pickle=False)
    # np.save(f'csvs/Bprops/0unwraps/heterogenous/linker_lengths_45or46_50nucs.npy', links45_46, allow_pickle=False)
    # qprop45_46 = wlc.bareWLC_gprop(kvals, links45_46, unwrap, props=Kprops_bareWLC)
    # integral45_46 = wlc.BRN_fourier_integrand_splines(kvals, links45_46, unwrap, Bprop=bprops45_46, rvals=rvals) #default: 1000 rvals
    # qintegral45_46 = wlc.BRN_fourier_integrand_splines(kvals, links45_46, unwrap, Bprop=qprop45_46, rvals=rvals) #default: 1000 rvals
    # #integral takes ~10 min to run, so prob worth saving
    # np.save(f'csvs/Bprops/{unwrap}unwraps/heterogenous/kinkedWLC_greens_links45or46_{len(rvals)}rvals_50nucs.npy', integral45_46, allow_pickle=False)
    # np.save(f'csvs/Bprops/{unwrap}unwraps/heterogenous/bareWLC_greens_links45or46_{len(rvals)}rvals_50nucs.npy', qintegral45_46, allow_pickle=False)
    # print(f'Saved G(R;L) for 45/46 link, {unwrap} unwrap!')



