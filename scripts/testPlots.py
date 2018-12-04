# -*- coding: utf-8 -*-
"""Worflow for analyzing chains with fixed linkers and unwrapping amounts:
1) Plot propgator in K-space, ensure curves look smooth and that points are capturing bumps.
2) Calculate G(R;L) using spline integration.
3) Plot looping probabilities vs. fragment length.
4) repeat 1-3 with bare WLC of same Rmax. """

import pickle
import numpy as np
#from matplotlib import pyplot as plt
import scipy
import pandas as pd
from pathlib import Path
from nuc_chain import fluctuations as wlc
from MultiPoint import propagator
from MultiPoint import WLCgreen
from scipy import stats

#K spacing used for all chains2
#TO CALCULATE MINIMUM linear K SPACING NEEDED in N=1 high K limit:
# period = 2pi / N, where N = L/(2lp)
#a bump is half a period, so one bump = pi * (2lp) / L = pi*300bp / linker length in bp (e.g. for 50bp bump is 20 K vals)

def compute_min_lin_k_spacing(link, unwrap, point_per_bump=6, Kmax=10**5):
    """Calculates minimum number of linearly spaced points from Kmin = 0 to Kmax so as to capture
    periodic bumps in B(K;L) with the resolution of points_per_bump. This calculates period based
    on rigid rod assumption (B(K;L) = sin(KN)/KN). So only valid for low N, aka if linker is long,
    this period won't be true. But also note that longer linkers fall off at lower Ks so log spacing
    should take care of them."""
    size_of_bump = np.pi * (2*wlc.default_lp) / link
    kspacing = size_of_bump / point_per_bump
    min_lin_spacing = Kmax / kspacing
    return min_lin_spacing

def save_greens_fixed_links_fixed_unwraps(links, unwrap=0, num_nucs=50, Kprops_bareWLC=None):
    """To process many chains with fixed linker lengths, load in the saved Bprops, do the Fourier
    inversion, and save the output so we can quickly look at them later."""
    Klin = np.linspace(0, 10**5, 20000)
    Klog = np.logspace(-3, 5, 10000)
    Kvals = np.unique(np.concatenate((Klin, Klog)))
    #convert to little k -- units of inverse bp (this results in kmax = 332)
    kvals = Kvals / (2*wlc.default_lp)
    if Kprops_bareWLC is None:
        Kprops_bareWLC = wlc.tabulate_bareWLC_propagators(Kvals)
    print('Tabulated K propagators for bare WLC!')

    for link in links:
        linkers = np.tile(link, num_nucs)
        #for each chain, do fourier inversion integral and save output as .npy file in Bprops directory
        Bprop = np.load(f'csvs/Bprops/{unwrap}unwraps/{link}link/B0_k_given_N_{link}bplinkers_{unwrap}unwraps_50nucs_30000Ks.npy')
        rvals = np.linspace(0.0, 1.0, 1000)
        qprop = wlc.bareWLC_gprop(kvals, linkers, unwrap, props=Kprops_bareWLC) #default: 1000 rvals
        integral = wlc.BRN_fourier_integrand_splines(kvals, linkers, unwrap, Bprop=Bprop, rvals=rvals) #default: 1000 rvals
        qintegral = wlc.BRN_fourier_integrand_splines(kvals, linkers, unwrap, Bprop=qprop, rvals=rvals) #default: 1000 rvals
        #integral takes ~10 min to run, so prob worth saving
        np.save(f'csvs/Bprops/{unwrap}unwraps/{link}link/kinkedWLC_greens_{link}link_{unwrap}unwraps_{len(rvals)}rvals_50nucs.npy', integral, allow_pickle=False)
        np.save(f'csvs/Bprops/{unwrap}unwraps/{link}link/bareWLC_greens_{link}link_{unwrap}unwraps_{len(rvals)}rvals_50nucs.npy', qintegral, allow_pickle=False)
        print(f'Saved G(R;L) for {link}link, {unwrap} unwrap!')

def save_greens_hetero_links_plus_one(links, unwrap=0, Kprops_bareWLC=None):
    """Each link in links is the smallest linker length in the heterogenous chain.
    Assumes each chain has random sampling of link or (link + 1)bp."""
    Klin = np.linspace(0, 10**5, 20000)
    Klog = np.logspace(-3, 5, 10000)
    Kvals = np.unique(np.concatenate((Klin, Klog)))
    #convert to little k -- units of inverse bp (this results in kmax = 332)
    kvals = Kvals / (2*wlc.default_lp)
    if Kprops_bareWLC is None:
        Kprops_bareWLC = wlc.tabulate_bareWLC_propagators(Kvals)
    print('Tabulated K propagators for bare WLC!')
    link = min(links)
    #for each chain, do fourier inversion integral and save output as .npy file in Bprops directory
    Bprop = np.load(f'csvs/Bprops/{unwrap}unwraps/heterogenous/B0_k_given_N_links{link}or{link+1}_50nucs_30000Ks.npy')
    rvals = np.linspace(0.0, 1.0, 1000)
    qprop = wlc.bareWLC_gprop(kvals, links, unwrap, props=Kprops_bareWLC) #default: 1000 rvals
    integral = wlc.BRN_fourier_integrand_splines(kvals, links, unwrap, Bprop=Bprop, rvals=rvals) #default: 1000 rvals
    qintegral = wlc.BRN_fourier_integrand_splines(kvals, links, unwrap, Bprop=qprop, rvals=rvals) #default: 1000 rvals
    #integral takes ~10 min to run, so prob worth saving
    np.save(f'csvs/Bprops/{unwrap}unwraps/heterogenous/kinkedWLC_greens_links{link}or{link+1}_{len(rvals)}rvals_50nucs.npy', integral, allow_pickle=False)
    np.save(f'csvs/Bprops/{unwrap}unwraps/heterogenous/bareWLC_greens_links{link}or{link+1}_{len(rvals)}rvals_50nucs.npy', qintegral, allow_pickle=False)
    print(f'Saved G(R;L) for {link} or {link+1} link, {unwrap} unwrap!')

### GROW CHAIN from 400 to 500 nucs to capture Gaussian chain limit ###
%%time
unwrap=0
Klin = np.linspace(0, 10**5, 20000)
Klog = np.logspace(-3, 5, 10000)
Kvals = np.unique(np.concatenate((Klin, Klog)))
#convert to little k -- units of inverse bp (this results in kmax = 332)
kvals = Kvals / (2*wlc.default_lp)
Kprops_bareWLC = wlc.tabulate_bareWLC_propagators(Kvals)
#calculate one per nuc for the first 30 nucs and then just calculate every other (long chain limit)
Nvals = np.arange(400, 501, 2)

links = np.tile(50, 500)
qprop50link_400to500nucs = wlc.bareWLC_gprop(kvals, links, 0, props=Kprops_bareWLC, Nvals=Nvals)
qintegral50link_400to500nucs = wlc.BRN_fourier_integrand_splines(kvals, links, unwrap, Bprop=qprop50link_400to500nucs, rvals=rvals, Nvals=Nvals)
np.save(f'csvs/Bprops/{unwrap}unwraps/{links[0]}link/bareWLC_greens_{links[0]}link_{unwrap}unwraps_{len(rvals)}rvals_400to500nucs.npy', qintegral50link_400to500nucs, allow_pickle=False)

links = np.tile(42, 500)
qprop42link_400to500nucs = wlc.bareWLC_gprop(kvals, links, 0, props=Kprops_bareWLC, Nvals=Nvals)
qintegral42link_400to500nucs = wlc.BRN_fourier_integrand_splines(kvals, links, unwrap, Bprop=qprop42link_400to500nucs, rvals=rvals, Nvals=Nvals)
np.save(f'csvs/Bprops/{unwrap}unwraps/{links[0]}link/bareWLC_greens_{links[0]}link_{unwrap}unwraps_{len(rvals)}rvals_400to500nucs.npy', qintegral42link_400to500nucs, allow_pickle=False)

links = np.tile(36, 500)
rvals = np.linspace(0.0, 1.0, 1000)
unwrap = 0
bprop36_400to500nucs = np.load(f'csvs/Bprops/{unwrap}unwraps/36link/B0_k_given_N_36bplinkers_0unwraps_400to500nucs_30000Ks.npy')
integral36_400to500nucs = wlc.BRN_fourier_integrand_splines(kvals, links, unwrap, Bprop=bprop36_400to500nucs, rvals=rvals, Nvals=Nvals)
np.save(f'csvs/Bprops/{unwrap}unwraps/{links[0]}link/kinkedWLC_greens_{links[0]}link_{unwrap}unwraps_{len(rvals)}rvals_400to500nucs.npy', integral36_400to500nucs, allow_pickle=False)

qprop36link_400to500nucs = wlc.bareWLC_gprop(kvals, links, 0, props=Kprops_bareWLC, Nvals=Nvals)
qintegral36link_400to500nucs = wlc.BRN_fourier_integrand_splines(kvals, links, unwrap, Bprop=qprop36link_400to500nucs, rvals=rvals, Nvals=Nvals)
np.save(f'csvs/Bprops/{unwrap}unwraps/{links[0]}link/bareWLC_greens_{links[0]}link_{unwrap}unwraps_{len(rvals)}rvals_400to500nucs.npy', qintegral36link_400to500nucs, allow_pickle=False)

def save_greens_100nuc_chains(links, unwrap=0):
    Klin = np.linspace(0, 10**5, 20000)
    Klog = np.logspace(-3, 5, 10000)
    Kvals = np.unique(np.concatenate((Klin, Klog)))
    #convert to little k -- units of inverse bp (this results in kmax = 332)
    kvals = Kvals / (2*wlc.default_lp)
    
    for link in links:
        linkers = np.tile(link, 50)
        Bprop_1to50 = pickle.load(open(f'csvs/Bprops/{unwrap}unwraps/{link}link/B0_k_given_N_{link}bplinkers_{unwrap}unwraps_50nucs_30000Ks.p', 'rb'))
        Bprop_50to100 = np.load(f'csvs/Bprops/{unwrap}unwraps/{link}link/B0_k_given_N_{link}bplinkers_{unwrap}unwraps_50to100nucs_30000Ks.p')
        #the last column of Bprop_1to50 and the first column of Bprop_50to100 are identical (both for 50 nucs)
        #75 points, one for each of the first 50 nucs, and every other from nucs50-100
        Nvals = np.concatenate(np.arange(1, 50), np.arange(50, 101, 2)) 
        Bprop_1to100nucs = np.concatenate((Bprop_50_1to50nucs, Bprop_50_50to100nucs[:, 1:]), axis=1) #shape = (29999, 75)
        rvals = np.linspace(0.0, 1.0, 1000)  
        integral = wlc.BRN_fourier_integrand_splines(kvals, linkers, unwrap, Bprop=Bprop_1to100nucs, rvals=rvals)

##### WORKFLOW FOR ANALYZING CHAIN #########
#assuming greens functions have already been saved

#load in all of the greens functions for following chains:
#0 unwraps: 35, 36, 37, 38, 42, 45, 46, 47, 50
#20 unwraps: 36, 42

links = np.array([35, 36, 37, 38, 42, 45, 46, 47, 50])
greens35_0unwraps_kinked = np.load('csvs/Bprops/0unwraps/35link/kinkedWLC_greens_35link_0unwraps_1000rvals_50nucs.npy')
greens35_0unwraps_bare = np.load('csvs/Bprops/0unwraps/35link/bareWLC_greens_35link_0unwraps_1000rvals_50nucs.npy')

greens36_0unwraps_kinked = np.load('csvs/Bprops/0unwraps/36link/kinkedWLC_greens_36link_0unwraps_1000rvals_50nucs.npy')
greens36_0unwraps_bare = np.load('csvs/Bprops/0unwraps/36link/bareWLC_greens_36link_0unwraps_1000rvals_50nucs.npy')

greens37_0unwraps_kinked = np.load('csvs/Bprops/0unwraps/37link/kinkedWLC_greens_37link_0unwraps_1000rvals_50nucs.npy')
greens37_0unwraps_bare = np.load('csvs/Bprops/0unwraps/37link/bareWLC_greens_37link_0unwraps_1000rvals_50nucs.npy')

greens38_0unwraps_kinked = np.load('csvs/Bprops/0unwraps/38link/kinkedWLC_greens_38link_0unwraps_1000rvals_50nucs.npy')
greens38_0unwraps_bare = np.load('csvs/Bprops/0unwraps/38link/bareWLC_greens_38link_0unwraps_1000rvals_50nucs.npy')

greens42_0unwraps_kinked = np.load('csvs/Bprops/0unwraps/42link/kinkedWLC_greens_42link_0unwraps_1000rvals_50nucs.npy')
greens42_0unwraps_bare = np.load('csvs/Bprops/0unwraps/42link/bareWLC_greens_42link_0unwraps_1000rvals_50nucs.npy')

greens45_0unwraps_kinked = np.load('csvs/Bprops/0unwraps/45link/kinkedWLC_greens_45link_0unwraps_1000rvals_50nucs.npy')
greens45_0unwraps_bare = np.load('csvs/Bprops/0unwraps/45link/bareWLC_greens_45link_0unwraps_1000rvals_50nucs.npy')

greens46_0unwraps_kinked = np.load('csvs/Bprops/0unwraps/46link/kinkedWLC_greens_46link_0unwraps_1000rvals_50nucs.npy')
greens46_0unwraps_bare = np.load('csvs/Bprops/0unwraps/46link/bareWLC_greens_46link_0unwraps_1000rvals_50nucs.npy')

greens47_0unwraps_kinked = np.load('csvs/Bprops/0unwraps/47link/kinkedWLC_greens_47link_0unwraps_1000rvals_50nucs.npy')
greens47_0unwraps_bare = np.load('csvs/Bprops/0unwraps/47link/bareWLC_greens_47link_0unwraps_1000rvals_50nucs.npy')

greens50_0unwraps_kinked = np.load('csvs/Bprops/0unwraps/50link/kinkedWLC_greens_50link_0unwraps_1000rvals_50nucs.npy')
greens50_0unwraps_bare = np.load('csvs/Bprops/0unwraps/50link/bareWLC_greens_50link_0unwraps_1000rvals_50nucs.npy')

greens36_20unwraps_kinked =(
np.load('csvs/Bprops/20unwraps/36link/kinkedWLC_greens_36link_20unwraps_1000rvals_50nucs.npy'))
greens36_20unwraps_bare =(
np.load('csvs/Bprops/20unwraps/36link/bareWLC_greens_36link_20unwraps_1000rvals_50nucs.npy'))


greens42_20unwraps_kinked =(
np.load('csvs/Bprops/20unwraps/42link/kinkedWLC_greens_42link_20unwraps_1000rvals_50nucs.npy'))
greens42_20unwraps_bare =(
np.load('csvs/Bprops/20unwraps/42link/bareWLC_greens_42link_20unwraps_1000rvals_50nucs.npy'))

#Parameters for plotting analysis
LINK = 
UNWRAP = 0
integral = greens37_0unwraps_kinked
qintegral = greens37_0unwraps_bare
# Klin = np.linspace(0, 10**5, 20000)
# Klog = np.logspace(-3, 5, 10000)
# Kvals = np.unique(np.concatenate((Klin, Klog)))
# #convert to little k -- units of inverse bp (this results in kmax = 332)
# kvals = Kvals / (2*wlc.default_lp)
# #suppose chain consists of 50, 36bp linkers -- this is about the length at which the R^2 plot levels out
links = np.tile(LINK, 50)
unwrap = UNWRAP

# #kinked propagator
# Bprop = pickle.load(open(f'csvs/Bprops/{unwrap}unwraps/{LINK}link/B0_k_given_N_{LINK}bplinkers_{unwrap}unwraps_{len(links)}nucs_30000Ks.p', 'rb'))
# #Quinn's propagator (q for Quinn!)
# Kprops_bareWLC = wlc.tabulate_bareWLC_propagators(Kvals)
# qprop = wlc.bareWLC_gprop_fixed_linkers_fixed_unwraps(kvals, links, unwrap, props=Kprops_bareWLC)

# #plot propagators vs. K --- make sure function looks smooth at high K, low N
# wlc.plot_BKN(Kvals, Bprop, links, unwrap)
# wlc.plot_BKN(Kvals, qprop, links, unwrap)

# #Plot G(R;L) for different subsets of chain lengths
lowNvals = np.arange(1, 6) #first 5 nucleosomes
highNvals = np.arange(10, 60, 10) #10-50 nucleosomes
#wlc.plot_greens(integral, links, unwrap, lowNvals)
#wlc.plot_greens(integral, links, unwrap, highNvals)

# #Compare against Bare WLC of same chain length
wlc.plot_greens_kinkedWLC_bareWLC(integral, qintegral, links, unwrap, lowNvals, rminN1=0.7, qrminN1=0.7)
wlc.plot_greens_kinkedWLC_bareWLC(integral, qintegral, links, unwrap, highNvals)

# #Looping probabilities
integrals = [greens35_0unwraps_kinked, greens36_0unwraps_kinked, greens37_0unwraps_kinked, greens38_0unwraps_kinked,
            greens42_0unwraps_kinked, greens45_0unwraps_kinked, greens46_0unwraps_kinked, 
            greens47_0unwraps_kinked, greens50_0unwraps_kinked]
labels = ['35bp', '36bp', '37bp', '38bp', '42bp', '45bp', '46bp', '47bp', '50bp']
plot_prob_loop_vs_fragment_length(integrals, labels, links, unwrap)

#Fit to Gaussian chain


#### TO ANALYZE NUMERICS OF SINGLE PROPAGATOR: 8/07/18 ####

LINK = 245
UNWRAP = 0
links = np.tile(LINK, 1)
unwrap = UNWRAP
Klin = np.linspace(0, 10**5, 20000)
Klog = np.logspace(-3, 5, 10000)
Kvals = np.unique(np.concatenate((Klin, Klog)))
#convert to little k -- units of inverse bp (this results in kmax = 332)
kvals = Kvals / (2*wlc.default_lp)
Bprop15 = np.load(f'csvs/Bprops/{unwrap}unwraps/{LINK}link/B0_k_given_{LINK}link_{unwrap}unwraps_1nuc_30000Ks.npy')
fig, ax = plt.subplots()
ax.loglog(Kvals, Bprop15.real)
plt.xlabel('K')
plt.ylabel('B(K;L)')
plt.title(f'$B_0^0(K;L)$ for {LINK}bp linkers, {unwrap}bp unwrapping')
L = 146 + LINK #total length
plt.legend([f'L={L:.0f}bp'])
plt.ylim([10**-12, 2])



