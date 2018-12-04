"""Script to generate figures for Beltran & Kannan et. al.

Two figures were made by hand. Figure 1 is a pair of blender renderings. The
relevant blend file names are simply mentioned below.

Where data has to be computed, this is mentioned."""
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import scipy
from scipy import stats
from pathlib import Path
from scipy.optimize import curve_fit
from nuc_chain import geometry as ncg
from nuc_chain import linkers as ncl
from MultiPoint import propagator
from nuc_chain import fluctuations as wlc
from nuc_chain.linkers import convert
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

# Plotting parameters
# PRL Font preference: computer modern roman (cmr), medium weight (m), normal shape
cm_in_inch = 2.54
# column size is 8.6 cm
col_size = 8.6 / cm_in_inch
inter_params = {
    'axes.edgecolor': 'black',
    'axes.grid': False,
    'axes.titlesize': 8.0,

    'axes.linewidth': 0.75,
    'backend': 'pdf',
    'axes.labelsize': 7,
    'legend.fontsize': 6,

    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'text.usetex': False,
    'figure.figsize': [col_size, col_size*(5/7)],

    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',

    'mathtext.fontset': 'stixsans',
    'savefig.format': 'pdf',
    'xtick.bottom':True,
    'xtick.major.pad': 2,
    'xtick.major.size': 4,
    'xtick.major.width': 0.5,

    'ytick.left':True,
    'ytick.right':False,
    'ytick.major.pad': 2,
    'ytick.major.size': 4,
    'ytick.major.width': 0.5,
    'ytick.minor.right':False,
    'lines.linewidth':1
}
medium_params = {
    'axes.edgecolor': 'black',
    'axes.grid': False,
    'axes.titlesize': 8.0,

    'axes.linewidth': 0.75,
    'backend': 'pdf',
    'axes.labelsize': 8.5,
    'legend.fontsize': 8,

    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'text.usetex': False,
    'figure.figsize': [col_size, col_size*(5/7)],

    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',

    'mathtext.fontset': 'stixsans',
    'savefig.format': 'pdf',
    'xtick.bottom':True,
    'xtick.major.pad': 2,
    'xtick.major.size': 4,
    'xtick.major.width': 0.5,

    'ytick.left':True,
    'ytick.right':False,
    'ytick.major.pad': 2,
    'ytick.major.size': 4,
    'ytick.major.width': 0.5,
    'ytick.minor.right':False,
    'lines.linewidth':1
}
plt.rcParams.update(inter_params)

small_params = {
    'axes.edgecolor': 'black',
    'axes.grid': False,
    'axes.titlesize': 8.0,

    'axes.linewidth': 0.75,
    'backend': 'pdf',
    'axes.labelsize': 6,
    'legend.fontsize': 6,

    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'text.usetex': False,
    'figure.figsize': [col_size, col_size*(5/7)],

    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',

    'mathtext.fontset': 'stixsans',
    'savefig.format': 'pdf',
    'xtick.bottom':True,
    'xtick.major.pad': 2,
    'xtick.major.size': 4,
    'xtick.major.width': 0.5,

    'ytick.left':True,
    'ytick.right':False,
    'ytick.major.pad': 2,
    'ytick.major.size': 4,
    'ytick.major.width': 0.5,
    'ytick.minor.right':False,
    'lines.linewidth':1
}
teal_flucts = '#387780'
red_geom = '#E83151'
dull_purple = '#755F80'

def plot_fig2a():
    """The r2 of the 36bp homogenous chain (0 unwrapping) compared to the
    wormlike chain with the corresponding Kuhn length."""
    plt.rcParams.update(inter_params)
    fig, ax = plt.subplots(figsize=(0.32*col_size, 0.21*col_size))
    hdf = pd.read_csv('./csvs/r2/r2-fluctuations-mu_36-sigma_0_10_0unwraps.csv')
    try:
        del hdf['Unnamed: 0']
    except:
        pass
    hdf = hdf.set_index(['variance', 'chain_id']).loc[0.0, 0.0]
    hdf.iloc[0,0] = 1
    hdf.iloc[0,1] = 1
    plt.plot(hdf['rmax'], hdf['r2'], color=dull_purple)
    x = np.logspace(0, 7, 100)
    y = wlc.r2wlc(x, hdf['kuhn'].mean()/2)
    plt.plot(x, y, '-.', color=teal_flucts, markersize=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([0.5, 2000])
    plt.ylim([0.5, 20000])
    plt.xlabel('Total Linker Length (nm)')
    plt.ylabel(r'$\sqrt{\langle R^2 \rangle}$')
    plt.legend([r'$L_i = 36bp$', r'$WLC, l_p \approx 3 nm$'],
               bbox_to_anchor=(0, 1.02, 1, .102), loc=3, borderaxespad=0)
    plt.savefig('plots/PRL/fig2a_r2_homogenous_vs_wlc.pdf', bbox_inches='tight')

# this ended up being supplemental after all
# def plot_fig2b():
#     plt.rcParams.update(medium_params)
#     links = np.load('csvs/linker_lengths_homogenous_so_far.npy')
#     kuhns = np.load('csvs/kuhns_homogenous_so_far.npy')
#     fig, ax = plt.subplots(figsize=(0.7*col_size, 0.42*col_size))
#     ax.plot(links, kuhns, '-o', markersize=1, lw=0.75, color=teal_flucts, label='Chromatin')
#     ax.plot(np.linspace(min(links), max(links), 1000), np.tile(100, 1000), '--', lw=0.75, label='Bare DNA', color=dull_purple)
#     plt.xlabel('Fixed linker length (bp)')
#     plt.ylabel('Kuhn length (nm)')
#     plt.xscale('log')
#     plt.legend(loc=4) #lower right
#     plt.subplots_adjust(left=0.19, bottom=0.28, top=0.98, right=0.99)
#     plt.tick_params(left=True, right=False, bottom=True, length=4)
#     plt.savefig('plots/PRL/fig2b_kuhn_length_homogenous_1to1000links_0unwraps.pdf')

def plot_fig2b():
    plt.rcParams.update(inter_params)
    kuhns = np.load('csvs/kuhns_1to250links_0to146unwraps.npy')
    fig, ax = plt.subplots(figsize=(0.45*col_size, 0.45*(5/7)*col_size))
    plt.xlabel('Fixed linker length (bp)')
    plt.ylabel('Kuhn length (nm)')
    links = np.arange(31, 52)
    ax.plot(links, kuhns[links-1, 0], '--o', markersize=4, lw=1.5, color=teal_flucts)
    plt.xticks(np.arange(31, 52, 2))
    plt.xlim([30, 48])
    plt.subplots_adjust(left=0.12, bottom=0.25, top=0.98, right=0.99)
    plt.tick_params(left=True, right=False, bottom=True, length=4)
    plt.savefig('plots/PRL/fig2b_kuhn_length_in_nm_31to51links_0unwraps.pdf')

#def plot_fig3_kuhn_length_vs_variance(sigmas=np.arange(0, 11), ax=None):
#    kuhnsf41 = np.load(f'csvs/r2/kuhns-fluctuations-mu41-sigma_0_10_0unwraps.npy')
#    #kuhnsf47 = np.load(f'csvs/r2/kuhns-fluctuations-mu47-sigma_0_10_0unwraps.npy')
#    kuhnsg41 = np.load(f'csvs/r2/kuhns-geometrical-mu41-sigma_0_10_0unwraps.npy')
#    #kuhnsg47 = np.load(f'csvs/r2/kuhns-geometrical-mu47-sigma_0_10_0unwraps.npy')
#    if ax is None:
#        fig, ax = plt.subplots(figsize=(0.95*col_size, 0.95*(5/7)*col_size))
#    #entire figure
#    ax.plot(sigmas, kuhnsg41, '--^', markersize=3, lw=1, label='Geometrical', color='#E83151')
#    ax.plot(sigmas, kuhnsf41, '-o', markersize=3, lw=1, label='Fluctuations', color='#387780')
#    #ax1.set_title('Mean linker length: 41bp')
#    #ax1.set_ylabel('Kuhn length (nm)')
#    #ax1.set_ylim([0, 200])
#    #ax1.tick_params(labelsize=18)
#    #plt.tick_params(labelsize=18)
#    # ax.plot(sigmas, kuhnsg47, '--^', markersize=2, lw=0.75, label='Geometrical', color=red_geom)
#    # ax.plot(sigmas, kuhnsf47, '-o', markersize=2, lw=0.75, label='Fluctuations', color=teal_flucts)
#    #ax.set_title('Mean linker length: 47bp')
#    plt.ylim([0, 200])
#    plt.legend()
#    plt.ylabel('Kuhn length (nm)')
#    plt.xlabel(r'Variance in linker length $\pm [x] bp$')
#    plt.subplots_adjust(left=0.14, bottom=0.15, top=0.98, right=0.99)
#    #plt.savefig('plots/PRL/fig4_kuhn_length_vs_window_size_mu47bp.pdf')

def plot_fig3(sigmas=np.arange(0, 41)):
    plt.rcParams.update(inter_params)
    fig, ax = plt.subplots(figsize=(0.95*col_size, 0.95*(5/7)*col_size))
    kuhnsf41_sig0to10 = np.load(f'csvs/r2/kuhns-fluctuations-mu41-sigma_0_10_0unwraps.npy')
    kuhnsf41_sig11to40 = np.load(f'csvs/r2/kuhns-fluctuations-mu41-sigma_11_40_0unwraps.npy')
    kuhnsf41 = np.concatenate((kuhnsf41_sig0to10, kuhnsf41_sig11to40))
    kuhnsg41_sig0to10 = np.load(f'csvs/r2/kuhns-geometrical-mu41-sigma_0_10_0unwraps.npy')
    kuhnsg41_sig11to40 = np.load(f'csvs/r2/kuhns-geometrical-mu41-sigma_11_40_0unwraps.npy')
    kuhnsg41 = np.concatenate((kuhnsg41_sig0to10, kuhnsg41_sig11to40))
    ax.plot(sigmas, kuhnsf41, '-o', markersize=3, label='Fluctuations',
            color=teal_flucts)

    ax.plot(sigmas, kuhnsg41, '--^', markersize=3, label='Geometrical',
            color=red_geom)
    rdf = pd.read_csv('./csvs/r2/r2-fluctuations-exponential-link-mu_41-0unwraps.csv')
    b = rdf['kuhn'].mean()
    xlim = plt.xlim()
    plt.plot([-10, 50], [b, b], 'k-.', label='Maximum Entropy')
    plt.xlim(xlim)
    ax.set_ylim([0, 100])
    plt.xlabel('$\sigma$ (bp)')
    plt.ylabel('Kuhn length (nm)')
    plt.legend()
    fig.text(1.3, 0, r'$\pm 0 bp$', size=9)
    fig.text(1.6, 0, r'$\pm 2 bp$', size=9)
    fig.text(1.9, 0, r'$\pm 6 bp$', size=9)
    plt.subplots_adjust(left=0.07, bottom=0.15, top=0.92, right=0.97)
    plt.savefig('./plots/PRL/fig-3-kuhn_length_vs_window_size_41_sigma0to40.pdf')

def plot_fig4a(ax=None):
    """The r2 of the 36bp homogenous chain (0 unwrapping) compared to the
    wormlike chain with the corresponding Kuhn length."""
    plt.rcParams.update(inter_params)
    fig, ax = plt.subplots(figsize=(0.32*col_size, 0.21*col_size))
    rdf = pd.read_csv('./csvs/r2/r2-fluctuations-exponential-link-mu_36-0unwraps.csv')
    try:
        del rdf['Unnamed: 0']
    except:
        pass
    for i, chain in rdf.groupby(['mu', 'chain_id']):
        chain.iloc[0,0] = 1
        chain.iloc[0,1] = 1
        plt.plot(chain['rmax'], chain['r2'], color=dull_purple, alpha=0.4)
        break
    x = np.logspace(0, 7, 100)
    y = wlc.r2wlc(x, rdf['kuhn'].mean()/2)
    plt.plot(x, y, '-', color='k')
    plt.legend([r'$\langle L_i \rangle= 36bp$', r'$WLC, l_p \approx 15 nm$'],
               bbox_to_anchor=(0, 1.02, 1, .102), loc=3, borderaxespad=0)
    for i, chain in rdf.groupby(['mu', 'chain_id']):
        chain.iloc[0,0] = 1
        chain.iloc[0,1] = 1
        plt.plot(chain['rmax'], chain['r2'], color=dull_purple, alpha=0.4)
    plt.plot(x, y, '-', color='k')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([0.5, 100000])
    plt.ylim([0.5, 10000000])
    plt.xlabel('Total Linker Length (nm)')
    plt.ylabel(r'$\sqrt{\langle R^2 \rangle}$')
    plt.savefig('plots/PRL/fig4a_r2_exp_vs_wlc.pdf', bbox_inches='tight')

def plot_fig4b(ax=None):
    plt.rcParams.update(inter_params)
    kuhnsf = np.load(f'csvs/r2/kuhns_exponential_fluctuations_mu2to180.npy')
    kuhnsg = np.load(f'csvs/r2/kuhns_exponential_geometrical_mu2to149.npy')
    kuhns_homo = np.load('csvs/kuhns_homogenous_so_far.npy')
    mug = np.load('csvs/r2/mus_exponential_geometrical.npy') #2 to 149
    mug = mug[1:99] #just plot mu from 3 to 100
    # kuhnsg = kuhnsg[1:99]
    kuhnsf = kuhnsf[1:99] #only plot first 149 points
    kuhns_homo = kuhns_homo[3:101]
    if ax is None:
        fig, ax = plt.subplots(figsize=(0.45*col_size, 0.45*(5/7)*col_size))
    #geometrical vs fluctuating
    #dashed line at 100 nm
    ax.plot(np.linspace(0, max(mug), 1000), np.tile(100, 1000), '--', lw=0.75,
            label='Bare WLC', color=teal_flucts)
    # ax.plot(mug[0:94], kuhnsg[0:94], '^', markersize=1, label='Geometrical', color=red_geom)
    ax.plot(mug, kuhnsf, label='Exponential', color=teal_flucts)
    #homogenous kuhn lengths faded in background
    ax.plot(mug, kuhns_homo, color=dull_purple, alpha=0.5, lw=0.75, label='Homogenous')


    #lines for yeast, mice, human
    yeast = 15
    mice = 45
    human = 56
    linelocs = [yeast, mice, human]
    # ax.text(yeast+2, 6, "A")
    # ax.text(mice+2, 6, "B")
    # ax.text(human+2, 6, "C")
    ax.vlines(linelocs, [0, 0, 0], [kuhnsf[yeast-3], kuhnsf[mice-3], kuhnsf[human-3]])
    #best fit line for geometrical case
    # m, b, rval, pval, stderr = stats.linregress(mug, kuhnsg)
    # best_fit = lambda x: m*x + b
    # xvals = np.linspace(51, 100, 40)
    # ax.plot(xvals, best_fit(xvals), ':', lw=0.75, color=red_geom)
    plt.ylim([0, 110])
    plt.legend(loc=(0.05, 0.6))
    plt.subplots_adjust(left=0.14, bottom=0.15, top=0.98, right=0.99)
    plt.xlabel(r'$\langle L_i \rangle$ (bp)')
    plt.ylabel(r'Kuhn length (nm)')
    plt.savefig('plots/PRL/fig4b_kuhn_exponential.pdf', bbox_inches='tight')


#for all Mayavi images
def plot_homogenous_chain(link, num_nucs):
    links = np.tile(link, num_nucs)
    chain = ncg.minimum_energy_no_sterics_linker_only(links, unwraps=0)
    ncg.visualize_chain(*chain, links, unwraps=0, plot_nucleosomes=False, plot_spheres=True, plot_exit=False)

def plot_MLC():
    entry_pos = np.loadtxt('csvs/MLC/r110v0')
    entry_us = np.loadtxt('csvs/MLC/u110v0')
    entry_t3 = entry_us[:, 0:3]
    entry_t2 = entry_us[:, 3:]
    entry_t1 = np.cross(entry_t2, entry_t3, axis=1)
    num_nucs = entry_pos.shape[0]
    entry_rots = []
    for i in range(num_nucs):
        #t1, t2, t3 as columns
        rot = np.eye(3)
        rot[:, 0] = entry_t1[i, :]
        rot[:, 1] = entry_t2[i, :]
        rot[:, 2] = entry_t3[i, :]
        entry_rots.append(rot)
    entry_rots = np.array(entry_rots)
    #skip the first nucleosome since entry_rot has NaNs
    entry_rots = entry_rots[1:, :, :]
    entry_pos = entry_pos[1:, :]
    return entry_rots, entry_pos
    #^above files saved in csvs/MLC in npy format

#Looping supplemental figure
#For looping main figure, see hetero31to52_loops_090418.py
def plot_homogenous_loops():
    kink41 = np.load(f'csvs/Bprops/0unwraps/41link/kinkedWLC_greens_41link_0unwraps_1000rvals_50nucs.npy')
    kink47 = np.load(f'csvs/Bprops/0unwraps/47link/kinkedWLC_greens_47link_0unwraps_1000rvals_50nucs.npy')
    bare41 = np.load(f'csvs/Bprops/0unwraps/41link/bareWLC_greens_41link_0unwraps_1000rvals_50nucs.npy')
    integrals = [kink47, kink41, bare41]
    labels = ['47bp', '41bp', 'Straight chain']
    links_list = [np.tile(47, 50), np.tile(41, 50), np.tile(41, 50)]
    plot_prob_loop_vs_fragment_length(integrals, labels, links_list, unwrap=0, nucmin=2)
    plt.subplots_adjust(left=0.19, bottom=0.21, top=0.96, right=0.97)
    plt.savefig('plots/loops/looping_homogenous_41_47_straight_chain.png')

def plot_prob_loop_vs_fragment_length(integrals, labels, links, unwrap, Nvals=None, nucmin=2, **kwargs):
    """Plot looping probability vs. chain length, where looping probability defined as G(0;L).

    Parameters
    ----------
    integrals : (L,) list of (rvals.size, Nvals.size) greens function arrays
        list of matrices G(r; N) where columns correspond to Nvals
    labels : (L,) array-like
        strings corresponding to label for each greens function (printed in legend)
    links : (L,) list of (num_linkers,) arrays
        list of full set of linkers in each chain, where num_linkers is the total number of
        nucleosomes in each chain
    unwrap : float
        unwrapping amount in bp. Assumes fixed unwrapping.
    Nvals : array-like
        number of linkers down the chain for which each green's functions in 'integrals' was calculated.
        Defaults to one per monomer of the chain. Assumes Nvals is the same for all chains for which
        you are plotting looping probabilities.
    nucmin : float
        minimum number of nucleosomes for which looping probability should be plotted. Defaults to 2,
        since first nucleosome is numerically not trusted. For shorter linkers (<42bp), recommended
        to set nucmin to 3 since first two points are sketchy.

    """

    if Nvals is None:
        Nvals = np.arange(1, len(links[0])+1)

    fig, ax = plt.subplots(figsize=(6.08, 3.84))
    #ignore first couple nucleosomes because of noise
    indmin = nucmin-1
    inds = Nvals - 1
    inds = inds[inds >= indmin]
    color_red = sns.color_palette("hls", 8)[0]
    #HARD CODE COLOR TUPLE: #D9A725 corresponds to
        #yellow = (217./255, 167./255, 37./255)
    #HARD CODE COLOR TUPE: #387780 corresponds to
        #teal = (56./255, 119./225, 128./255)
    colors = [color_red, '#D9A725', '#387780']
    print(colors)
    for i in range(len(labels)):
        ldna = convert.genomic_length_from_links_unwraps(links[i], unwraps=unwrap)
        ploops = integrals[i][0, indmin:]
        pldna = ldna[inds]
        ax.loglog(pldna, ploops, '-o', markersize=2, linewidth=1,
            color=colors[i], label=labels[i], **kwargs)
    ax.legend(loc=(0.32, 0.03), frameon=False)
    plt.xlabel('Genomic distance (bp)')
    plt.ylabel('$P_{loop}$ ($bp^{-3}$)')
    #plt.title(f'Looping probability vs. Chain Length')
    plt.tick_params(left=True, right=False, bottom=True)
    return fig, ax

def calculate_kuhn_length_from_r2(df, mu, chain_length, **kwargs):
    """Calculate :math:`b=\langle{R^2}\rangle/R_{max}` in the long chain
    limit (roughly 5000 monomers down the chain)."""

    df2 = df.sort_values('rmax')
    kuhns = []
    for var, vals in df2.groupby(['variance']):
        sample_links = ncl.fake_linkers_increasing_variance(mu, var, size=(chain_length-1,), type='box')
        sample_rmax = convert.Rmax_from_links_unwraps(sample_links, **kwargs)
        #Assume long chain limit is 5000 monomers down a random chain sampled from this distribution.
        min_rmax_for_kuhn = sample_rmax[5000] * ncg.dna_params['lpb']
        rmax_long = vals.rmax[vals['rmax']>=min_rmax_for_kuhn]
        r2_long = vals.r2[vals['rmax']>=min_rmax_for_kuhn]
        kuhns.append(stats.linregress(rmax_long, r2_long)[0])
    return np.array(kuhns)

#Save kuhn lengths as npy files so I don't have to git annex the huge csv's
def extract_kuhn_lengths_from_r2():
    links = [41, 47]
    for link in links:
        # dffsig10 = pd.read_csv(f'csvs/r2/r2-fluctuations-mu_{link}-sigma_0_10_0unwraps.csv')
        # kuhnfsig10 = calculate_kuhn_length_from_r2(dffsig10, link, 7500, unwraps=0)
        # np.save(f'csvs/r2/kuhns-fluctuations-mu{link}-sigma_0_10_0unwraps.npy', kuhnfsig10)
        # dfgsig10 = pd.read_csv(f'csvs/r2/r2-geometrical-mu_{link}-sigma_0_10_0unwraps.csv')
        # kuhngsig10 = calculate_kuhn_length_from_r2(dfgsig10, link, 7500, unwraps=0)
        # np.save(f'csvs/r2/kuhns-geometrical-mu{link}-sigma_0_10_0unwraps.npy', kuhngsig10)
        dffsig11to20 = pd.read_csv(f'csvs/r2/r2-fluctuations-mu_{link}-sigma_11_20_0unwraps.csv')
        kuhnfsig11to20 = calculate_kuhn_length_from_r2(dffsig11to20, link, 7500, unwraps=0)
        dffsig20to30 = pd.read_csv(f'csvs/r2/r2-fluctuations-mu_{link}-sigma_20_30_0unwraps.csv')
        kuhngsig11to30 = np.concatenate((kuhnfsig11to20, calculate_kuhn_length_from_r2(dffsig20to30, link, 7500, unwraps=0)))
        dffsig31to40 = pd.read_csv(f'csvs/r2/r2-fluctuations-mu_{link}-sigma_31_40_0unwraps.csv')
        kuhnfsig11to40 = np.concatenate((kuhngsig11to30, calculate_kuhn_length_from_r2(dffsig31to40, link, 7500, unwraps=0)))
        np.save(f'csvs/r2/kuhns-fluctuations-mu{link}-sigma_11_40_0unwraps.npy', kuhnfsig11to40)
        #still waiting on sigmas 11 onwards; afterwards, concatenate sigmas 0 - 40



