r"""
Deepti Kannan
11-20-18

Tabulated green's functions for heterogenous chains which sample from exponentially
distributed linkers with mu=56bp. So far, have calculated 19 chains of 100 nucs and
57 chains of 50 nucs for a total of 76 chains.

This script loads in the greens functions from all the above chains, calculates looping
probabilities as a function of genomic distance, saves this data in a csv, computes rolling
average of these points, and plots individual chain configurations with average.

"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from MultiPoint import propagator
from nuc_chain import fluctuations as wlc
from nuc_chain import geometry as ncg
from nuc_chain.linkers import convert
from pathlib import Path
from scipy import stats
import seaborn as sns
from multiprocessing import Pool
from functools import partial
from scipy import interpolate

params = {'axes.edgecolor': 'black', 'axes.facecolor': 'white', 'axes.grid': False, 'axes.titlesize': 18.0,
'axes.linewidth': 0.75, 'backend': 'pdf','axes.labelsize': 18,'legend.fontsize': 18,
'xtick.labelsize': 14,'ytick.labelsize': 14,'text.usetex': False,'figure.figsize': [7, 5],
'mathtext.fontset': 'stixsans', 'savefig.format': 'pdf', 'xtick.bottom':True, 'xtick.major.pad': 5, 'xtick.major.size': 5, 'xtick.major.width': 0.5,
'ytick.right':True, 'ytick.major.pad': 5, 'ytick.major.size': 5, 'ytick.major.width': 0.5, 'ytick.minor.right':False, 'ytick.minor.left':False, 'lines.linewidth':2}

plt.rcParams.update(params)

def compute_looping_statistics_heterogenous_chains(nucmin=2):
    """Compute and save looping probabilities for all 'num_chains' heterogenous chains
    saved in the links31to52 directory.
    """
    indmin = nucmin-1
    #directory in which all chains are saved
    dirpath = Path('csvs/Bprops/0unwraps/heterogenous/exp_mu56')
    #Create one data frame per chain and add to this list; concatenate at end
    list_dfs = []

    #first load in chains of length 100 nucs
    chainIDs_100nucs = np.concatenate(([1, 2], np.arange(101, 118)))
    for j in chainIDs_100nucs:
        df = pd.DataFrame(columns=['num_nucs', 'chain_id', 'ldna', 'rmax', 'ploops'])
        chaindir = f'100nucs_chain{j}'
        links = np.load(dirpath/chaindir/f'linker_lengths_{chaindir}_100nucs.npy')
        greens = np.load(dirpath/chaindir/f'kinkedWLC_greens_{chaindir}_100nucs.npy')
        #only including looping statistics for 2 nucleosomes onwards when plotting though
        df['ldna'] = convert.genomic_length_from_links_unwraps(links, unwraps=0)[indmin:]
        df['rmax'] = convert.Rmax_from_links_unwraps(links, unwraps=0)[indmin:]
        df['ploops'] = greens[0, indmin:]
        df['num_nucs'] = 100
        df['chain_id'] = j
        df['chaindir'] = chaindir
        list_dfs.append(df)

    #next load in chains of length 50 nucs
    chainIDs_50nucs = np.concatenate((np.arange(11, 15), np.arange(21, 23), np.arange(41, 44), [57, 65],
                                      np.arange(101, 130), np.arange(131, 139), np.arange(141, 150)))
    for j in chainIDs_50nucs:
        df = pd.DataFrame(columns=['num_nucs', 'chain_id', 'ldna', 'rmax', 'ploops'])
        chaindir = f'50nucs_chain{j}'
        links = np.load(dirpath/chaindir/f'linker_lengths_{chaindir}_50nucs.npy')
        greens = np.load(dirpath/chaindir/f'kinkedWLC_greens_{chaindir}_50nucs.npy')
        #only including looping statistics for 2 nucleosomes onwards when plotting though
        df['ldna'] = convert.genomic_length_from_links_unwraps(links, unwraps=0)[indmin:]
        df['rmax'] = convert.Rmax_from_links_unwraps(links, unwraps=0)[indmin:]
        df['ploops'] = greens[0, indmin:]
        df['num_nucs'] = 50
        df['chain_id'] = j
        df['chaindir'] = chaindir
        list_dfs.append(df)

    #Concatenate list into one data frame
    df = pd.concat(list_dfs, ignore_index=True, sort=False)
    df.to_csv(dirpath/'looping_probs_heterochains_exp_mu56_0unwraps.csv')
    return df

def plot_looping_probs_hetero_avg(df, **kwargs):
    #df2 = df.sort_values('ldna')
    fig, ax = plt.subplots(figsize=(7.21, 5.19))
    #first just plot all chains
    palette = sns.cubehelix_palette(n_colors=np.unique(df['chaindir']).size)
    #palette = sns.color_palette("husl", np.unique(df['chaindir']).size)
    sns.lineplot(data=df, x='ldna', y='ploops', hue='chaindir', palette=palette,
        legend=None, ci=None, ax=ax, alpha=0.5, lw=1)

    #Then plot running average
    df2 = df.sort_values('ldna')
    df3 = df2.drop(columns=['chaindir'])
    df4 = df3.rolling(100).mean()
    #df4.plot(x='ldna', y='ploops', legend=False, color='k', linewidth=3, ax=ax, label='Average')

    #try plotting average of linear interpolations
    # xvals = np.linspace(np.min(df.ldna), np.max(df.ldna), 1000)
    # dflin = pd.DataFrame(columns=['chaindir', 'ldna', 'ploops'])
    # for i, dfs in df.groupby('chaindir'):
    #     f = interpolate.interp1d(dfs.ldna, dfs.ploops)
    #     ax.plot(xvals, f(xvals), linewidth=1)
        #dflin[]
        #dfs.plot(x='ldna', y='ploops', legend=False, color=palette[i], linewidth=1, ax=ax)

    #plot power law scaling
    xvals = np.linspace(4675, 9432, 1000)
    gaussian_prob = 10**-1.4*np.power(xvals, -1.5)
    ax.loglog(xvals, gaussian_prob, 'k')
    #vertical line of triangle
    ax.vlines(9432, gaussian_prob[-1], gaussian_prob[0])
    #print(gaussian_prob[0])
    #print(gaussian_prob[-1])
    ax.hlines(gaussian_prob[0], 4675, 9432)
    #ax.text(9500, 8.4*10**(-8), "-3", fontsize=18)
    ax.text(7053.5, 10**(-6.5), '$L^{-3/2}$', fontsize=18)

    #compare to bare WLC with constant linker length 56 bp
    indmin = 1
    bare41 = np.load('csvs/Bprops/0unwraps/41link/bareWLC_greens_41link_0unwraps_1000rvals_50nucs.npy')
    ldna = convert.genomic_length_from_links_unwraps(np.tile(41, 50), unwraps=0)
    ax.loglog(ldna[indmin:], bare41[0, indmin:], '--',
            color='#387780', label='Straight chain', **kwargs)

    #plot gaussian probability from analytical kuhn length calculation
    #Kuhn length for mu = 56, exponential (in nm)
    # b = 40.662 / ncg.dna_params['lpb']
    # #b =
    # analytical_gaussian_prob = (3.0 / (2*np.pi*df2['rmax']*b))**(3/2)
    # ax.loglog(df2['ldna'], analytical_gaussian_prob, ':', label='Gaussian chain with $b=40.66$nm')
    plt.legend(loc=4)
    plt.xlabel('Genomic distance (bp)')
    plt.ylabel('$P_{loop}$ ($bp^{-3}$)')
    plt.title(f'Exponentially distributed linkers mu=56bp')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([10**-12.5, 10**-5.5])
    plt.tick_params(left=True, right=False, bottom=True)
    plt.subplots_adjust(left=0.15, bottom=0.16, top=0.91, right=0.95)
    #return df4
    plt.savefig('plots/loops/looping_exp_mu56_vs_bareWLC_76chains.png')

def fit_persistance_length_to_gaussian_looping_prob(df4, ldna_min=4675):
    """Fit effective persistance length to log-log looping probability vs. chain length (Rmax).
    Nmin is the minimum number of nucleosomes to begin powerlaw fitting to Gaussian chain.
    Takes in data frame with data from heterogenous chains. Assumes ldna values are sorted and
    rolling average has been taken.

    Parameters
    ----------
    ldna_min : float
        minimum number of basepairs down the chain to begin linear fit.
    """
    ploops = df4['ploops']
    ldna = df4['ldna']
    Rmax = df4['rmax']

    #Gaussian chain limit in log-log space
    ploop_gaussian = np.log(ploops[ldna >= ldna_min])
    Rmax_gaussian = np.log(Rmax[ldna >= ldna_min])
    m, intercept, rvalue, pvalue, stderr = stats.linregress(Rmax_gaussian, ploop_gaussian)
    print(f'Power law: L^{m}')

    #For Guassian chain, the intercept = (3/2)log(3/(4pi*lp)) -- see Deepti's notes
    lp = 3 / (4*np.pi*np.exp(intercept/np.abs(m)))
    lp = lp * ncg.dna_params['lpb']
    return m, lp

