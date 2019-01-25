r"""
Deepti Kannan
09-04-18

Tabulated green's functions for heterogenous chains which sample uniformly from linkers
31-51bp inclusive (corresponds to one period). So far, have calculated 5 chains of 100 nucs,
50 chains of 50 nucs, and 30 chains of 25ish nucs (at least 4675bp of DNA).

This script loads in the greens functions from all the above chains, calculates looping
probabilities as a function of genomic distance, saves this data in a csv, computes rolling
average of these points, and plots everything nicely in a way that's presentable at
group meeting.

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

"""All variables needed for analysis"""
Klin = np.linspace(0, 10**5, 20000)
Klog = np.logspace(-3, 5, 10000)
Kvals = np.unique(np.concatenate((Klin, Klog)))
#convert to little k -- units of inverse bp (this results in kmax = 332)
kvals = Kvals / (2*wlc.default_lp)

def compute_looping_statistics_heterogenous_chains(nucmin=2):
    """Compute and save looping probabilities for all 'num_chains' heterogenous chains
    saved in the links31to52 directory.
    """
    indmin = nucmin-1
    #directory in which all chains are saved
    dirpath = Path('csvs/Bprops/0unwraps/heterogenous/links31to52')
    #Create one data frame per chain and add to this list; concatenate at end
    list_dfs = []

    # #first load in chains of length 100 nucs
    # for j in range(1, 6):
    #     df = pd.DataFrame(columns=['num_nucs', 'chain_id', 'ldna', 'rmax', 'ploops'])
    #     chaindir = f'100nucs_chain{j}'
    #     links = np.load(dirpath/chaindir/f'linker_lengths_{chaindir}_100nucs.npy')
    #     greens = np.load(dirpath/chaindir/f'kinkedWLC_greens_{chaindir}_100nucs.npy')
    #     #only including looping statistics for 2 nucleosomes onwards when plotting though
    #     df['ldna'] = convert.genomic_length_from_links_unwraps(links, unwraps=0)[indmin:]
    #     df['rmax'] = convert.Rmax_from_links_unwraps(links, unwraps=0)[indmin:]
    #     df['ploops'] = greens[0, indmin:]
    #     df['num_nucs'] = 100
    #     df['chain_id'] = j
    #     df['chaindir'] = chaindir
    #     list_dfs.append(df)

    #next load in chains of length 50 nucs
    for j in range(1, 51):
        df = pd.DataFrame(columns=['num_nucs', 'chain_id', 'ldna', 'rmax', 'ploops'])
        chaindir = f'50nucs_chain{j}'
        #named these files incorrectly
        if j==11 or (j>=13 and j <=20):
            wrong_chaindir = f'100nucs_chain{j}'
            links = np.load(dirpath/chaindir/f'linker_lengths_{wrong_chaindir}_50nucs.npy')
            greens = np.load(dirpath/chaindir/f'kinkedWLC_greens_{wrong_chaindir}_50nucs.npy')
        else:
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

    #finally, load in chains of length 25ish nucs
    for j in np.concatenate((np.arange(1, 18), np.arange(21, 28))):
        df = pd.DataFrame(columns=['num_nucs', 'chain_id', 'ldna', 'ploops'])
        chaindir = f'25nucs_chain{j}'
        links = np.load(dirpath/chaindir/f'linker_lengths_{chaindir}_25nucs.npy')
        greens = np.load(dirpath/chaindir/f'kinkedWLC_greens_{chaindir}_25nucs.npy')
        #only including looping statistics for 2 nucleosomes onwards when plotting though
        df['ldna'] = convert.genomic_length_from_links_unwraps(links, unwraps=0)[indmin:]
        df['rmax'] = convert.Rmax_from_links_unwraps(links, unwraps=0)[indmin:]
        df['ploops'] = greens[0, indmin:]
        df['num_nucs'] = 25
        df['chain_id'] = j
        df['chaindir'] = chaindir
        list_dfs.append(df)

    #Concatenate list into one data frame
    df = pd.concat(list_dfs, ignore_index=True, sort=False)
    df.to_csv(dirpath/'looping_probs_heterochains_links31to52_0unwraps.csv')
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
    df4 = df3.rolling(75).mean()
    df4.plot(x='ldna', y='ploops', legend=False, color='k', linewidth=3, ax=ax, label='Average')

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

    #compare to bare WLC
    indmin = 1
    bare41 = np.load('csvs/Bprops/0unwraps/41link/bareWLC_greens_41link_0unwraps_1000rvals_50nucs.npy')
    ldna = convert.genomic_length_from_links_unwraps(np.tile(41, 50), unwraps=0)
    ax.loglog(ldna[indmin:], bare41[0, indmin:], '--',
            color='#387780', label='Straight chain', **kwargs)

    #plot gaussian probability from analytical kuhn length calculation
    #Kuhn length for mu = 41, box variance = 10 (in nm)
    b = 27.525 / ncg.dna_params['lpb']
    #b =
    analytical_gaussian_prob = (3.0 / (2*np.pi*df4['rmax']*b))**(3/2)
    ax.loglog(df4['ldna'], analytical_gaussian_prob, ':', label='Gaussian chain with $b=27.5$nm')
    plt.legend()
    plt.xlabel('Genomic distance (bp)')
    plt.ylabel('$P_{loop}$ ($bp^{-3}$)')
    plt.title(f'Uniformly random linkers 31-51bp')
    plt.xscale('log')
    plt.yscale('log')
    plt.tick_params(left=True, right=False, bottom=True)
    plt.subplots_adjust(left=0.15, bottom=0.16, top=0.91, right=0.95)
    return df4
    #plt.savefig('plots/loops/looping_hetero31to52bp_vs_bareWLC.png')

def plot_gaussian_kinkedAverage_intersection(df4, ldna_min=4675):
    fig, ax = plt.subplots(figsize=(7.21, 5.19))
    #fit average power law and persistence length from rolling average of hetero chains
    # ldna_vals = np.logspace(10**3, 10**7, 10000)
    # m_fit, lp_fit = fit_persistance_length_to_gaussian_looping_prob(df4, ldna_min)
    # #convert to bp
    # lp_fit = lp_fit / ncg.dna_params['lpb']
    # intercept_kinked = 1.5*np.log(3/(4*np.pi*lp_fit))
    # #plot gaussian probability from analytical kuhn length calculation
    # #Kuhn length for mu = 41, box variance = 10 (in nm)
    # m = -1.5
    # b = 27.525 / ncg.dna_params['lpb']
    # lp = b/2
    # #b =
    # intercept_gaussian = 1.5*np.log(3/(4*np.pi*lp))
    # ax.plot(ldna_vals, np.exp(m_fit*np.log(ldna_vals) + intercept_kinked), 'k', label='Average kinked chain')
    # ax.plot(ldna_vals, np.exp(m*np.log(ldna_vals) + intercept_gaussian), ':', label='Gaussian chain with $b=27.5$nm')
    df4.plot(x='ldna', y='ploops', legend=False, color='k', ax=ax, label='Average')
    b = 27.525 / ncg.dna_params['lpb']
    #b =
    analytical_gaussian_prob = (3.0 / (2*np.pi*df4['rmax']*b))**(3/2)
    ax.loglog(df4['ldna'], analytical_gaussian_prob, ':', label='Gaussian chain with $b=27.5$nm')
    plt.legend()
    plt.xlabel('Genomic distance (bp)')
    plt.ylabel('$P_{loop}$ ($bp^{-3}$)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([10**3, 10**4])
    plt.tick_params(left=True, right=False, bottom=True)
    plt.subplots_adjust(left=0.15, bottom=0.16, top=0.91, right=0.95)


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

