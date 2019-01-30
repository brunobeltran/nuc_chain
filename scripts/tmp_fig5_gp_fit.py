import re
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from nuc_chain.linkers import convert


def compute_looping_statistics_heterogenous_chains():
    """Compute and save looping probabilities for all 'num_chains' heterogenous chains
    saved in the links31to52 directory.
    """
    #directory in which all chains are saved
    dirpath = Path('csvs/Bprops/0unwraps/heterogenous/exp_mu56')
    #Create one data frame per chain and add to this list; concatenate at end
    list_dfs = []
    #first load in chains of length 100 nucs
    file_re = re.compile("([0-9]+)nucs_chain([0-9]+)")
    for chain_folder in dirpath.glob('*'):
        match = file_re.match(chain_folder.name)
        if match is None:
            continue
        num_nucs, chain_id = match.groups()
        try:
            links = np.load(chain_folder
                    /f'linker_lengths_{num_nucs}nucs_chain{chain_id}_{num_nucs}nucs.npy')
            greens = np.load(chain_folder
                    /f'kinkedWLC_greens_{num_nucs}nucs_chain{chain_id}_{num_nucs}nucs.npy')
        except FileNotFoundError:
            continue
        df = pd.DataFrame(columns=['num_nucs', 'chain_id', 'ldna', 'rmax', 'ploops'])
        #only including looping statistics for 2 nucleosomes onwards when plotting though
        df['ldna'] = convert.genomic_length_from_links_unwraps(links, unwraps=0)
        df['rmax'] = convert.Rmax_from_links_unwraps(links, unwraps=0)
        df['ploops'] = greens[0,:]
        df['num_nucs'] = num_nucs
        df['chain_id'] = chain_id
        list_dfs.append(df)
    #Concatenate list into one data frame
    df = pd.concat(list_dfs, ignore_index=True, sort=False)
    df = df.set_index(['num_nucs', 'chain_id']).sort_index()
    df.to_csv(dirpath/'looping_probs_heterochains_exp_mu56_0unwraps.csv')
    return df

df = compute_looping_statistics_heterogenous_chains()
# if the first step is super short, we are numerically unstable
df.loc[df['rmax'] <= 5, 'ploops'] = np.nan
# if the output is obviously bad numerics, ignore it
df.loc[df['ploops'] < 10**(-13), 'ploops'] = np.nan
df = df.dropna()
df['log_ploop'] = np.log10(df['ploops'])
df['log_rmax'] = np.log10(df['rmax'])
df = df.sort_values('log_rmax')
# get an estimator of the variance from a rolling window
# window size chosen by eye
rolled = df.rolling(50).apply(np.nanmean, raw=False)
rolled_std = df.rolling(50).apply(np.nanstd, raw=False)
x = np.atleast_2d(df['log_rmax'].values.copy()).T
y = df['log_ploop'].values.copy().ravel()
sigma = np.interp(x, rolled['log_rmax'].values,
        rolled_std['log_ploop'].values,
        left=rolled_std['log_ploop'].values[0],
        right=rolled_std['log_ploop'].values[-1])
sigma = sigma.ravel()
# pandas rolling std doesn't like left boundary, but we can just fill in
# soemthing reasonable
sigma[np.isnan(sigma)] = np.nanmax(sigma)
# now fit to a gaussian process
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma, n_restarts_optimizer=10)
# gp.fit(x, y)
# pickle.dump(gp, open('gp_rmax.pkl', 'wb'))
