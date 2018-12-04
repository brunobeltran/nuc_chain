r"""
Deepti Kannan
08-27-18

This script will calculate the matrix of looping probabilities between
nearby nucleosomes near a methylation cite. Using the following dyad
positions extracted from Chereji data:
. . . 35, 23, 15, 26, 30, 245, nucleation site, 47, 21, 18, 15, 20, 17, 35 . . .
"""

import numpy as np
#from matplotlib import pyplot as plt
from MultiPoint import propagator
from nuc_chain import fluctuations as wlc
from nuc_chain import geometry as ncg
from pathlib import Path
from scipy import stats
#import seaborn as sns
from multiprocessing import Pool
from functools import partial

# params = {'axes.edgecolor': 'black', 'axes.facecolor': 'white', 'axes.grid': False, 
# 'axes.linewidth': 0.75, 'backend': 'pdf','axes.labelsize': 12,'legend.fontsize': 10,
# 'xtick.labelsize': 10,'ytick.labelsize': 10,'text.usetex': False,'figure.figsize': [5.75, 4.5], 
# 'mathtext.fontset': 'stixsans', 'savefig.format': 'pdf', 'xtick.major.pad': 4, 'xtick.major.size': 5, 'xtick.major.width': 0.5,
# 'ytick.major.pad': 4, 'ytick.major.size': 5, 'ytick.major.width': 0.5,}

# plt.rcParams.update(params)

# def links_from_dyad_locations(dyads, w_ins=ncg.default_w_in, w_outs=ncg.default_w_out,
#                                      helix_params=ncg.helix_params_best, unwraps=None):
# 	b = helix_params['b']
# 	num_linkers = len(dyads) - 1
# 	w_ins, w_outs = ncg.resolve_wrapping_params(unwraps, w_ins, w_outs, num_linkers+1)
# 	mu_ins = (b - 1)/2 - w_ins
# 	mu_outs = (b - 1)/2 - w_outs
# 	links = np.zeros((num_linkers,))
# 	dyad_to_dyad_distances = np.diff(dyads)
# 	for i, dyad_to_dyad in enumerate(dyad_to_dyad_distances):
# 		#bare linker length is dyad_to_dyad distance minus wrapped and unwrapped amounts
# 		links[i] = dyad_to_dyad - w_outs[i] - mu_outs[i] - mu_ins[i+1] - w_ins[i+1]
# 	return links


"""All variables needed for analysis"""
Klin = np.linspace(0, 10**5, 20000)
Klog = np.logspace(-3, 5, 10000)
Kvals = np.unique(np.concatenate((Klin, Klog)))
#convert to little k -- units of inverse bp (this results in kmax = 332)
kvals = Kvals / (2*wlc.default_lp)
links35 = np.tile(35, 44)
Rlinks = np.array([47, 21, 18, 15, 20, 17])
Llinks = np.array([245, 30, 26, 15, 23, 35])

#links to right of methylation site (50 in total)
Rlinks = np.concatenate((Rlinks, links35))
#links to left of methylation site (50 in total)
Llinks = np.concatenate((Llinks, links35))
unwrap = 0


if __name__ == '__main__':
    i=1
    #Calculate Bprops for chain to the right from (i+1) to 50 nucs
    print(f'Calculating Rlinks_{i+1}to50nucs...')
    filepath = Path(f'csvs/Bprops/0unwraps/heterogenous/Sarah/Rlinks_{i+1}to50nucs')
    with Pool(31) as pool:
        pool.map(partial(wlc.Bprop_k_given_L, links=Rlinks[i:], unwraps=0,
                   filepath=filepath), kvals)
    wlc.combine_Bprop_files(filepath, Rlinks[i:], unwraps=0)
    print(f'Calculating Llinks_{i+1}to50nucs...')
    filepath = Path(f'csvs/Bprops/0unwraps/heterogenous/Sarah/Llinks_{i+1}to50nucs')
    with Pool(31) as pool:
        pool.map(partial(wlc.Bprop_k_given_L, links=Llinks[i:], unwraps=0,
                   filepath=filepath), kvals)
    wlc.combine_Bprop_files(filepath, Llinks[i:], unwraps=0)







