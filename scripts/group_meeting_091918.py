#plotting parameters
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from nuc_chain import geometry as ncg
from nuc_chain import linkers as ncl
from nuc_chain.linkers import convert

#These follow Andy's plotting preferences for
params = {'axes.edgecolor': 'black', 'axes.facecolor': 'white', 'axes.grid': False, 'axes.titlesize': 18.0,
'axes.linewidth': 0.75, 'backend': 'pdf','axes.labelsize': 18,'legend.fontsize': 18,
'xtick.labelsize': 14,'ytick.labelsize': 14,'text.usetex': False,'figure.figsize': [7, 5],
'mathtext.fontset': 'stixsans', 'savefig.format': 'pdf', 'xtick.bottom':True, 'xtick.major.pad': 5, 'xtick.major.size': 5, 'xtick.major.width': 0.5,
'ytick.left':True, 'ytick.right':False, 'ytick.major.pad': 5, 'ytick.major.size': 5, 'ytick.major.width': 0.5, 'ytick.minor.right':False, 'lines.linewidth':2}

plt.rcParams.update(params)

def plot_kuhn_length_colored_by_period():
    fig, ax = plt.subplots(figsize=(12, 5))
    kuhns1to250 = np.load('csvs/kuhns_1to250links_0to146unwraps.npy')
    plt.plot(np.arange(1, 251), kuhns1to250[:,0], linewidth=1, color='#387780')
    plt.scatter(np.arange(1, 251), kuhns1to250[:,0], c=np.arange(1.5,251.5)%10.5, cmap='Spectral')
    plt.xlabel('Linker length (bp)')
    plt.ylabel('Kuhn Length (nm)')
    plt.title(r'$b=\langle{R^2}\rangle/R_{max}$, colored by (linker length % 10.5bp)')
    plt.colorbar()
    plt.tick_params(left=True, right=False, bottom=True, labelsize=16)
    plt.subplots_adjust(bottom=0.14, right=1.0, top=0.92)
    plt.savefig('plots/kuhn/kuhn_length_in_nm_1to250links_0unwraps.png')

def plot_kuhn_length_one_period_in_nm():
    kuhns = np.load('csvs/kuhns_1to250links_0to146unwraps.npy')
    fig, ax = plt.subplots(figsize=(6.27, 2.83))
    plt.xlabel('Linker length (bp)')
    plt.ylabel('Kuhn length (nm)')
    links = np.arange(31, 52)
    ax.plot(links, kuhns[links-1, 0], '-o', markersize=8, lw=3, color='#387780')
    plt.xticks(np.arange(31, 52, 2))
    plt.subplots_adjust(left=0.07, bottom=0.15, top=0.92, right=0.97)
    plt.tick_params(left=False, right=False, bottom=True, labelsize=16)
    plt.savefig('plots/kuhn/kuhn_length_in_nm_31to51links_0unwraps.png')

def plot_kuhn_length_one_period_36to56():
    kuhns = np.load('csvs/kuhns_1to250links_0to146unwraps.npy')
    fig, ax = plt.subplots(figsize=(6.27, 2.83))
    plt.xlabel('Linker length (bp)')
    plt.ylabel('Kuhn length (nm)')
    links = np.arange(36, 58)
    ax.plot(links, kuhns[links-1, 0], '-o', markersize=8, lw=3, color='#387780')
    plt.xticks(np.arange(36, 58, 2))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.subplots_adjust(left=0.12, bottom=0.15, top=0.92, right=0.97)
    plt.tick_params(left=True, right=False, bottom=True, labelsize=18)
    plt.savefig('plots/kuhn/kuhn_length_in_nm_31to51links_0unwraps.png')

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
    #^above files saved in csvs/MLC in npy format

#Looping supplemental figure
#For looping main figure, see hetero31to52_loops_090418.py
def plot_homogenous_loops():
    kink41 = np.load(f'csvs/Bprops/0unwraps/41link/kinkedWLC_greens_41link_0unwraps_1000rvals_50nucs.npy')
    kink47 = np.load(f'csvs/Bprops/0unwraps/47link/kinkedWLC_greens_47link_0unwraps_1000rvals_50nucs.npy')
    bare41 = np.load(f'csvs/Bprops/0unwraps/41link/bareWLC_greens_41link_0unwraps_1000rvals_50nucs.npy')
    integrals = [kink41, kink47, bare41]
    labels = ['41bp', '47bp', 'bare WLC']
    plot_prob_loop_vs_fragment_length(kinked_integrals, labels, links_list, unwrap=0, nucmin=2)
    plt.subplots_adjust(left=0.16, bottom=0.17, top=0.9, right=0.97)
    plt.savefig('plots/loops/looping_homogenous_41_47_bareWLC.png')

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

    fig, ax = plt.subplots(figsize=(6.17, 4.13))
    #ignore first couple nucleosomes because of noise
    indmin = nucmin-1
    inds = Nvals - 1
    inds = inds[inds >= indmin]
    colors = sns.color_palette("hls", 3)
    for i in range(len(labels)):
        ldna = convert.genomic_length_from_links_unwraps(links[i], unwraps=unwrap)
        ploops = integrals[i][0, indmin:]
        pldna = ldna[inds]
        ax.loglog(pldna, ploops, '-o', markersize=2, linewidth=1,
            color=colors[i], label=labels[i], **kwargs)
    ax.legend()
    plt.xlabel('Genomic distance (bp)')
    plt.ylabel('$P_{loop}$ ($bp^{-3}$)')
    plt.title(f'Looping probability vs. Chain Length')
    plt.tick_params(left=True, right=False, bottom=True)
    return fig, ax

def plot_greens_kinkedWLC_bareWLC(integral, qintegral, links, unwrap, Nvals, rvals=None, rminN1=0.0, qrminN1=0.0):
    """Plot G(R;Rmax) for kinked WLC and bare WLC with same Rmax vs. dimensionless chain length r = R/Rmax,
    one curve per Nval, where N is the chain length in number of nucleosomes.

    Parameters
    ----------
    integral : (rvals.size, Rmax.size) array-like
        Green's function for kinked WLC with this set of linkers and unwrapping
    qintegral : (rvals.size, Rmax.size) array-like
        Green's function for bare WLC with this set of linkers and unwrapping
    links : (L,) array-like
        bare linker length in bp (not including unwrapping)
    unwraps : float
        unwrapping amount in bp. Assumes fixed unwrapping.
    Nvals : array-like
        number of linkers down the chain for which you want to plot G(R;Rmax).
    rminN1 : float
        minimum r value from which the N=1 curve should be plotted for kinked WLC. Due to numerical
        issues, there tends to be noise for r values < 0.7. To avoid plotting this noise,
        set rminN1=0.7 (or whatever value seems fitting for your particular chain).
    qrminN1 : float
        minimum r value for which the N=1 curve should be plotted for bare WLC. e.g. qrminN1=0.7
        Note: hard-coded rmin to be 0.4 for chains of length N=2 because there tends to be noise
        for small r even for the N=2 case.
    """

    if rvals is None:
        rvals = np.linspace(0.0, 1.0, 1000)
    fig, ax = plt.subplots(figsize=(6.17, 4.13))
    ldna = ncg.genomic_length_from_links_unwraps(links, unwraps=unwrap)
    inds = Nvals - 1
    Ls = ldna[inds]

    for ii, i in enumerate(inds):
        color = np.random.rand(3)
        rmin = 0.0
        qrmin = 0.0
        if (i==0): #for N=1 case, don't plot noise
            rmin = rminN1
            qrmin = qrminN1
        if (i==1):
            qrmin = 0.4
        rsub = rvals[(rvals >= rmin)]
        qrsub = rvals[(rvals >= qrmin)]
        intsub = integral[(rvals >= rmin), i]
        qintsub = qintegral[(rvals >= qrmin), i]
        ax.semilogy(rsub, intsub, '-o', markersize=2, linewidth=1,
            color=color, label=f'L={Ls[ii]:.0f}bp, kinked')
        ax.semilogy(qrsub, qintsub, '--', color=color, label=f'L={Ls[ii]:.0f}bp, no kinks')

    ax.legend(frameon=True)
    plt.xlabel('$R/R_{max}$')
    plt.ylabel('G(R;L)')
    #plt.legend([f'L={L:.0f}bp' for L in Ls], frameon=True)
    plt.title(f'{links[0]}bp linkers, {unwrap} unwraps')
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

def plot_kuhn_length_exponential(ax=None):
    kuhnsf = np.load(f'csvs/r2/kuhns_exponential_fluctuations_mu2to180.npy')
    kuhnsg = np.load(f'csvs/r2/kuhns_exponential_geometrical_mu2to149.npy')
    mug = np.load('csvs/r2/mus_exponential_geometrical.npy') #2 to 149
    mug = mug[1:99] #just plot mu from 3 to 100
    kuhnsg = kuhnsg[1:99]
    kuhnsf = kuhnsf[1:99] #only plot first 149 points
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5.29))
    #geometrical vs fluctuating
    ax.plot(mug[0:94], kuhnsg[0:94], '^', markersize=6, label='Geometrical', color='#E83151')
    ax.plot(mug, kuhnsf, 'o', markersize=6, label='Fluctuations', color='#387780')
    #dashed line at 100 nm
    ax.plot(np.linspace(0, max(mug), 1000), np.tile(100, 1000), '--', lw=3, label='Bare WLC', color='#387780')

    #lines for yeast, mice, human
    yeast = 15
    ax.plot(np.linspace(0, yeast, 100), np.tile(kuhnsf[yeast-3], 100), '--', color='k', lw=2)
    mice = 45
    ax.plot(np.linspace(0, mice, 100), np.tile(kuhnsf[mice-3], 100), '--', color='k', lw=2)
    human = 56
    ax.plot(np.linspace(0, human, 100), np.tile(kuhnsf[human-3], 100), '--', color='k', lw=2)
    #best fit line for geometrical case
    m, b, rval, pval, stderr = stats.linregress(mug, kuhnsg)
    best_fit = lambda x: m*x + b
    xvals = np.linspace(51, 100, 40)
    ax.plot(xvals, best_fit(xvals), ':', lw=3, color='#E83151')
    plt.ylabel('Kuhn Length (nm)')
    plt.xlabel('Mean linker length (bp)')
    plt.ylim([0, 120])
    plt.legend()
    plt.tick_params(labelsize=16)
    plt.subplots_adjust(left=0.12, bottom=0.13, top=0.96, right=0.98)
    plt.savefig('plots/kuhn/kuhn_exponential_geom_vs_fluct_mu3to100.png')

def plot_kuhn_length_vs_variance(sigmas=np.arange(0, 11), ax=None):
    kuhnsf41 = np.load(f'csvs/r2/kuhns-fluctuations-mu41-sigma_0_10_0unwraps.npy')
    kuhnsf47 = np.load(f'csvs/r2/kuhns-fluctuations-mu47-sigma_0_10_0unwraps.npy')
    kuhnsg41 = np.load(f'csvs/r2/kuhns-geometrical-mu41-sigma_0_10_0unwraps.npy')
    kuhnsg47 = np.load(f'csvs/r2/kuhns-geometrical-mu47-sigma_0_10_0unwraps.npy')
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False, figsize=(12.69, 5.29))
    #entire figure
    ax1.plot(sigmas, kuhnsg41, '--^', markersize=8, label='Geometrical', color='#E83151')
    ax1.plot(sigmas, kuhnsf41, '-o', markersize=8, label='Fluctuations', color='#387780')
    ax1.set_title('mu=41bp')
    ax1.set_ylabel('Kuhn Length (nm)')
    ax1.set_ylim([0, 200])
    ax1.tick_params(labelsize=16)
    plt.tick_params(labelsize=16)
    ax2.plot(sigmas, kuhnsg47, '--^', markersize=8, label='Geometrical', color='#E83151')
    ax2.plot(sigmas, kuhnsf47, '-o', markersize=8, label='Fluctuations', color='#387780')
    ax2.set_title('mu=47bp')
    ax2.set_ylim([0, 200])
    ax2.tick_params(labelsize=16)
    plt.legend()
    fig.text(0.5, 0.04, r'Window Size $\pm [x] bp$', ha='center', size=18)
    plt.subplots_adjust(left=0.07, bottom=0.15, top=0.92, right=0.97)
    plt.savefig('plots/kuhn/kuhn_length_vs_window_size_41_47.png')

def plot_kuhn_length_vs_boxsize(sigmas=np.arange(0, 41)):
    #kuhnsf41 = np.load(f'csvs/r2/kuhns-fluctuations-mu41-sigma_0_10_0unwraps.npy')
    #kuhnsf47 = np.load(f'csvs/r2/kuhns-fluctuations-mu47-sigma_0_10_0unwraps.npy')
    #kuhnsg41 = np.load(f'csvs/r2/kuhns-geometrical-mu41-sigma_0_10_0unwraps.npy')
    #kuhnsg47 = np.load(f'csvs/r2/kuhns-geometrical-mu47-sigma_0_10_0unwraps.npy')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False, figsize=(12.69, 5.29))
    #entire figure
    ax1.plot(sigmas, kuhnsg41, '--^', markersize=8, label='Geometrical', color='#E83151')
    ax1.plot(sigmas, kuhnsf41, '-o', markersize=8, label='Fluctuations', color='#387780')
    ax1.set_title('mu=41bp')
    ax1.set_ylabel('Kuhn Length (nm)')
    ax1.set_ylim([0, 100])
    ax1.tick_params(labelsize=16)

    ax2.plot(sigmas, kuhnsg47, '--^', markersize=8, label='Geometrical', color='#E83151')
    ax2.plot(sigmas, kuhnsf47, '-o', markersize=8, label='Fluctuations', color='#387780')
    ax2.set_title('mu=47bp')
    ax2.set_ylim([0, 100])
    ax2.tick_params(labelsize=16)
    plt.legend()
    #fig.text(0.5, 0.04, r'Window Size $\pm [x] bp$', ha='center', size=18)
    plt.subplots_adjust(left=0.07, bottom=0.15, top=0.92, right=0.97)
    #plt.savefig('plots/kuhn/kuhn_leng_vs_window_size_41_47_mu0to40.png')

#for all Mayavi images
def plot_homogenous_chain(link, num_nucs):
    links = np.tile(link, num_nucs)
    chain = ncg.minimum_energy_no_sterics_linker_only(links, unwraps=0)
    ncg.visualize_chain(*chain, links, unwraps=0, plot_nucleosomes=False, plot_spheres=True, plot_exit=False)



