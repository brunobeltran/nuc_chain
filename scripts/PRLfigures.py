"""Script to generate figures for Beltran & Kannan et. al.

Two figures were made by hand. Figure 1 is a pair of blender renderings. The
relevant blend file names are simply mentioned below.

Where data has to be pre-computed, the procedure is mentioned."""
import re
from pathlib import Path
import pickle

import matplotlib.cm as cm
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from nuc_chain import geometry as ncg
from nuc_chain import linkers as ncl
from MultiPoint import propagator
from nuc_chain import fluctuations as wlc
from nuc_chain.linkers import convert

# Plotting parameters
# PRL Font preference: computer modern roman (cmr), medium weight (m), normal shape
cm_in_inch = 2.54
# column size is 8.6 cm
col_size = 8.6 / cm_in_inch
default_width = 1.0*col_size
aspect_ratio = 4/7
default_height = aspect_ratio*default_width
plot_params = {
    'backend': 'pdf',
    'savefig.format': 'pdf',
    'text.usetex': True,
    'font.size': 7,

    'figure.figsize': [default_width, default_height],
    'figure.facecolor': 'white',

    'axes.grid': False,
    'axes.edgecolor': 'black',
    'axes.facecolor': 'white',

    'axes.titlesize': 8.0,
    'axes.labelsize': 8,
    'legend.fontsize': 6.5,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'axes.linewidth': 0.75,

    'xtick.top': False,
    'xtick.bottom': True,
    'xtick.direction': 'out',
    'xtick.minor.size': 3,
    'xtick.minor.width': 0.5,
    'xtick.major.pad': 2,
    'xtick.major.size': 5,
    'xtick.major.width': 1,

    'ytick.left': True,
    'ytick.right': False,
    'ytick.direction': 'out',
    'ytick.minor.size': 3,
    'ytick.minor.width': 0.5,
    'ytick.major.pad': 2,
    'ytick.major.size': 5,
    'ytick.major.width': 1,

    'lines.linewidth': 1
}
plt.rcParams.update(plot_params)

teal_flucts = '#387780'
red_geom = '#E83151'
dull_purple = '#755F80'
rich_purple = '#e830e8'

def render_chain(linkers, unwraps=0, **kwargs):
        entry_rots, entry_pos = ncg.minimum_energy_no_sterics_linker_only(linkers, unwraps=unwraps)
        # on linux, hit ctrl-d in the ipython terminal but don't accept the
        # "exit" prompt to get the mayavi interactive mode to work. make sure
        # to use "off-screen rendering" and fullscreen your window before
        # saving (this is actually required if you're using a tiling window
        # manager like e.g. i3 or xmonad).
        ncg.visualize_chain(entry_rots, entry_pos, linkers, unwraps=unwraps, plot_spheres=True)

def draw_triangle(alpha, x0, width, orientation, base=10,
                            **kwargs):
    """Draw a triangle showing the best-fit slope on a linear scale.

    Parameters
    ----------
    alpha : float
        the slope being demonstrated
    x0 : (2,) array_like
        the "left tip" of the triangle, where the hypotenuse starts
    width : float
        horizontal size
    orientation : string
        'up' or 'down', control which way the triangle's right angle "points"
    base : float
        scale "width" for non-base 10

    Returns
    -------
    corner : (2,) np.array
        coordinates of the right-angled corner of the triangle
    """
    x0, y0 = x0
    x1 = x0 + width
    y1 = y0 + alpha*(x1 - x0)
    plt.plot([x0, x1], [y0, y1], 'k')
    if (alpha >= 0 and orientation == 'up') \
    or (alpha < 0 and orientation == 'down'):
        plt.plot([x0, x1], [y1, y1], 'k')
        plt.plot([x0, x0], [y0, y1], 'k')
        # plt.plot lines have nice rounded caps
        # plt.hlines(y1, x0, x1, **kwargs)
        # plt.vlines(x0, y0, y1, **kwargs)
        corner = [x0, y1]
    elif (alpha >= 0 and orientation == 'down') \
    or (alpha < 0 and orientation == 'up'):
        plt.plot([x0, x1], [y0, y0], 'k')
        plt.plot([x1, x1], [y0, y1], 'k')
        # plt.hlines(y0, x0, x1, **kwargs)
        # plt.vlines(x1, y0, y1, **kwargs)
        corner = [x1, y0]
    else:
        raise ValueError(r"Need $\alpha\in\mathbb{R} and orientation\in{'up', 'down'}")
    return corner

def draw_power_law_triangle(alpha, x0, width, orientation, base=10,
                            **kwargs):
    """Draw a triangle showing the best-fit power-law on a log-log scale.

    Parameters
    ----------
    alpha : float
        the power-law slope being demonstrated
    x0 : (2,) array_like
        the "left tip" of the power law triangle, where the hypotenuse starts
    width : float
        horizontal size in number of major log ticks (default base-10)
    orientation : string
        'up' or 'down', control which way the triangle's right angle "points"
    base : float
        scale "width" for non-base 10

    Returns
    -------
    corner : (2,) np.array
        coordinates of the right-angled corner of the triangle
    """
    x0, y0 = x0
    x1 = x0*base**width
    y1 = y0*(x1/x0)**alpha
    plt.plot([x0, x1], [y0, y1], 'k')
    if (alpha >= 0 and orientation == 'up') \
    or (alpha < 0 and orientation == 'down'):
        plt.plot([x0, x1], [y1, y1], 'k')
        plt.plot([x0, x0], [y0, y1], 'k')
        # plt.plot lines have nice rounded caps
        # plt.hlines(y1, x0, x1, **kwargs)
        # plt.vlines(x0, y0, y1, **kwargs)
        corner = [x0, y1]
    elif (alpha >= 0 and orientation == 'down') \
    or (alpha < 0 and orientation == 'up'):
        plt.plot([x0, x1], [y0, y0], 'k')
        plt.plot([x1, x1], [y0, y1], 'k')
        # plt.hlines(y0, x0, x1, **kwargs)
        # plt.vlines(x1, y0, y1, **kwargs)
        corner = [x1, y0]
    else:
        raise ValueError(r"Need $\alpha\in\mathbb{R} and orientation\in{'up', 'down'}")
    return corner

default_lis = [36, 38]
default_colors = [red_geom, rich_purple]
def plot_fig2a(lis=default_lis, colors=None):
    """The r2 of the 36bp homogenous chain (0 unwrapping) compared to the
    wormlike chain with the corresponding Kuhn length."""
    if colors is None:
        if len(lis) == 2:
            colors = default_colors
        else:
            colors = len(lis) * [red_geom]
    assert(len(colors) == len(lis))
    fig, ax = plt.subplots(figsize=(default_width, default_height))
    x = np.logspace(0, 7, 100)
    y = np.sqrt(wlc.r2wlc(x, 100))
    plt.plot(x, y, '.', color=[0,0,0], markersize=1)
    hdfs = {}
    for i, li in enumerate(lis):
        hdfs[li] = pd.read_csv(f'./csvs/r2/r2-fluctuations-mu_{li}-sigma_0_10_0unwraps.csv')
        try:
            del hdfs[li]['Unnamed: 0']
        except:
            pass
        hdfs[li] = hdfs[li].set_index(['variance', 'chain_id']).loc[0.0, 0.0]
        hdfs[li].iloc[0,0:2] = 1 # rmax,r2 == (0,0) ==> (1,1)
        plt.plot(hdfs[li]['rmax'], np.sqrt(hdfs[li]['r2']), color=colors[i])
    for li in lis:
        y = np.sqrt(wlc.r2wlc(x, hdfs[li]['kuhn'].mean()/2))
        plt.plot(x, y, '-.', color=teal_flucts, markersize=1)

    xmin = 1
    ymin = xmin
    ymax = 700
    xmax = 3_000
    # bands representing different regimes of the R^2
    plt.fill_between(x, ymin, ymax, where=x<12, color=[0.96, 0.95, 0.95])
    plt.fill_between(x, ymin, ymax, where=((x>=12)&(x<250)), color=[0.99, 0.99, 0.99])
    plt.fill_between(x, ymin, ymax, where=x>=250, color=[0.9, 0.9, 0.91])

    # power law triangle for the two extremal regimes
    corner = draw_power_law_triangle(1, [2, 3], 0.5, 'up')
    plt.text(3, 11, '$L^1$')
    corner = draw_power_law_triangle(1/2, [350, 30], 0.8, 'down')
    plt.text(700, 16, '$L^{1/2}$')

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Total linker length (nm)')
    plt.ylabel(r'End-to-end distance (nm)')
    legend = [r'Bare DNA'] \
           + [r'$L_i = ' + str(li) + r'$ bp' for li in lis] \
           + [r'WLC, best fit']
    plt.legend(legend)
    plt.tight_layout()
    plt.savefig('./plots/PRL/fig2a_r2_homogenous_vs_wlc.pdf', bbox_inches='tight')

def plot_fig2b():
    kuhns = np.load('csvs/kuhns_1to250links_0to146unwraps.npy')
    fig, ax = plt.subplots(figsize=(default_width, default_height))
    links = np.arange(31, 52)
    ax.plot(links, kuhns[links-1, 0], '--o', markersize=4, lw=1.5, color=teal_flucts)
    for i, li in enumerate(default_lis):
        ax.plot(li, kuhns[li-1, 0], '--o', markersize=4, color=default_colors[i])
    plt.xticks(np.arange(31, 50, 2))
    plt.xlim([31, 49])
    plt.xlabel('Fixed linker length (bp)')
    plt.ylabel('Kuhn length (nm)')
    plt.tight_layout()
    plt.savefig('plots/PRL/fig2b_kuhn_length_in_nm_31to51links_0unwraps.pdf')

def render_fig2b_chains(**kwargs):
    for li in [36, 38, 41, 47]:
        render_chain(14*[li], **kwargs)

def plot_fig3(sigmas=np.arange(0, 41)):
    fig, ax = plt.subplots(figsize=(default_width, default_height))
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

def render_fig3_chains(mu=41, sigmas=[0, 2, 6]):
    for sigma in sigmas:
        sign_bit = 2*np.round(np.random.rand(N)) - 1
        render_chain(mu + sign_bit*np.random.randint(sigma+1), size=(N,))

def plot_fig4b():
    fig, ax = plt.subplots(figsize=(default_width, default_height))
    kuhns = pd.read_csv('csvs/kuhns_so_far.csv')
    kuhns = kuhns.set_index(['variance_type', 'type', 'mu', 'variance'])
    mu_max = 100
    # dotted line at 100 nm
    ax.plot(np.linspace(0, mu_max, 100), np.tile(100, 100), '.',
            markersize=1, label='Bare WLC', color=[0,0,0])
    def make_plottable(df):
        df = df.groupby('mu').mean().reset_index()
        df = df[df['mu'] < mu_max].dropna()
        return df
    exp_fluct = kuhns.loc['exponential', 'fluctuations']
    exp_fluct = make_plottable(exp_fluct)
    ax.plot(exp_fluct['mu'], exp_fluct['b'], label='Exponential', color=teal_flucts)
    homo_fluct = kuhns.loc['homogenous', 'fluctuations']
    homo_fluct = make_plottable(homo_fluct)
    ax.plot(homo_fluct['mu'], homo_fluct['b'], color=dull_purple, alpha=0.5, lw=0.75, label='Homogenous')


    #lines for yeast, mice, human
    yeast = 15
    mice = 45
    human = 56
    linelocs = [yeast, mice, human]
    # ax.text(yeast+2, 6, "A")
    # ax.text(mice+2, 6, "B")
    # ax.text(human+2, 6, "C")
    ax.vlines(linelocs, [0, 0, 0], [exp_fluct.loc[exp_fluct['mu'] == loc, 'b'].values for loc in linelocs])
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

def plot_fig5(df=None, rmax_or_ldna='rmax', named_sim='mu56'):
    fig, ax = plt.subplots(figsize=(default_width, default_height))
    n = rmax_or_ldna
    logn = 'log_' + n
    # first set sim-specific parameters, draw scaling triangles at manually
    # chosen locations
    if (named_sim, rmax_or_ldna) == ('mu56', 'ldna'):
        draw_triangle(-3/2, x0=[3.5, -7], width=0.4, orientation='up')
        plt.text(3.6, -6.8, '$L^{-3/2}$')
        # manually set thresholds to account for numerical instability at low n
        min_logn = 2.5
    elif (named_sim, rmax_or_ldna) == ('mu56', 'rmax'):
        draw_triangle(-3/2, x0=[3.0, -7.5], width=0.4, orientation='up')
        plt.text(3.1, -7.3, '$L^{-3/2}$')
        min_logn = 2.2
    elif (named_sim, rmax_or_ldna) == ('links31-to-52', 'rmax'):
        draw_triangle(-3/2, x0=[3.0, -7.5], width=0.4, orientation='up')
        plt.text(3.1, -7.3, '$L^{-3/2}$')
        min_logn = 2.0
    elif (named_sim, rmax_or_ldna) == ('links31-to-52', 'ldna'):
        draw_triangle(-3/2, x0=[3.5, -7], width=0.4, orientation='up')
        plt.text(3.6, -6.8, '$L^{-3/2}$')
        min_logn = 2.5
    if df is None:
        df = load_looping_statistics_heterogenous_chains(named_sim=named_sim)
    # if the first step is super short, we are numerically unstable
    df.loc[df['rmax'] <= 5, 'ploops'] = np.nan
    # if the output is obviously bad numerics, ignore it
    df.loc[df['ploops'] < 10**(-13), 'ploops'] = np.nan
    df = df.dropna()
    df['log_ploop'] = np.log10(df['ploops'])
    df[logn] = np.log10(df[n])
    df = df.sort_values(logn)
    # get an estimator of the variance from a rolling window
    # window size chosen by eye
    rolled = df.rolling(50).apply(np.nanmean, raw=False)
    rolled_std = df.rolling(50).apply(np.nanstd, raw=False)
    x = np.atleast_2d(df[logn].values.copy()).T
    y = df['log_ploop'].values.copy().ravel()
    sigma = np.interp(x, rolled[logn].values,
            rolled_std['log_ploop'].values,
            left=rolled_std['log_ploop'].values[0],
            right=rolled_std['log_ploop'].values[-1])
    sigma = sigma.ravel()
    # pandas rolling std doesn't like left boundary, but we can just fill in
    # soemthing reasonable
    sigma[np.isnan(sigma)] = np.nanmax(sigma)
    # now fit to a gaussian process
    if Path(f'csvs/gp_{named_sim}_{rmax_or_ldna}.pkl').exists():
        gp = pickle.load(open(f'csvs/gp_{named_sim}_{rmax_or_ldna}.pkl', 'rb'))
    else:
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma**2, n_restarts_optimizer=10)
        gp.fit(x, y)
        pickle.dump(gp, open(f'gp_{named_sim}_{rmax_or_ldna}.pkl', 'wb'))
    xgrid = np.linspace(max(min_logn, np.min(x)), np.max(x), 100).reshape(-1, 1)
    y_pred, sig = gp.predict(xgrid, return_std=True)
    # plot all the individual chains, randomly chop some down to make plot look
    # nicer
    palette = sns.cubehelix_palette(n_colors=len(df.groupby(['num_nucs', 'chain_id'])))
    ord = np.random.permutation(len(palette))
    for i, (label, chain) in enumerate(df.groupby(['num_nucs', 'chain_id'])):
        num_nucs = int(label[0])
        max_nuc_to_plot = num_nucs*(1 - 0.2*np.random.rand())
        chain = chain[chain['nuc_id'] <= max_nuc_to_plot]
        chain = chain[chain[logn] >= min_logn]
        plt.plot(chain[logn].values, chain['log_ploop'].values,
                 c=palette[ord[i]], alpha=0.25, lw=0.5, label=None)
    # bold a couple of the chains
    bold_c = palette[int(9*len(palette)/10)]
    if named_sim == 'mu56':
        chains_to_bold = [1, 105, 112]
    elif named_sim == 'links31-to-52':
        chains_to_bold = [1, 3, 5]
    for chain_id in chains_to_bold:
        chain = df.loc[100, chain_id]
        chain = chain[chain[logn] >= min_logn]
        plt.plot(chain[logn].values, chain['log_ploop'].values, c=bold_c, alpha=0.8,
                 label=None)
    # 95% confidence intervals of mean estimate
    fill = plt.fill(np.concatenate([xgrid, xgrid[::-1]]),
             np.concatenate([y_pred - 1.9600 * sig, (y_pred + 1.9600 * sig)[::-1]]),
             alpha=.75, fc='r', ec='None', label='Average $\pm$ 95\%')
    # now get the best fit gaussian chain. chain becomes approximate gaussian
    # after about 1000rmax units, hence the 3 below
    m, intercept, rvalue, pvalue, stderr = stats.linregress(
            xgrid[xgrid>3].ravel(), y_pred[xgrid.ravel()>3])
    #For Guassian chain, the intercept = (3/2)log(3/(4pi*lp)) -- see Deepti's notes
    lp = 3 / (4*np.pi*10**(intercept/np.abs(m)))
    lp = lp * ncg.dna_params['lpb']
    # m is the power law exponent, should be -3/2

    # load in the straight chain, in [bp] (b = 100nm/ncg.dna_params['lpb'])
    bare_n, bare_ploop = wlc.load_WLC_looping()
    b = 2*wlc.default_lp
    min_bare_logn = 2.0
    lx = np.log10(bare_n)
    ly = np.log10(bare_ploop)
    l100 = plt.plot(lx[lx>=min_bare_logn], ly[lx>=min_bare_logn], '-.', c=teal_flucts,
             label=f'Straight chain, b=100nm')
    # now rescale the straight chain to match average
    if named_sim == 'mu56':
        b = 40.67 # nm
        k = b/100 # scaling between straight and 56bp exponential chain
        nn = 146/56 # wrapped amount to linker length ratio
    elif named_sim == 'links31-to-52':
        b = 2*13.762 # nm
        k = b/100 # scaling between straight and uniform chain
        nn = 146/41.5
    if rmax_or_ldna == 'ldna':
        # we use the fact that (e.g. for exp_mu56, 0 unwraps)
        # df['ldna'] = df['rmax'] + 146*df['nuc_id']
        # (on ave)   = df['rmax'] + 146*df['rmax']/56
        bare_n = bare_n*(1 + nn)
    lnx = np.log10(bare_n*k)
    lny = np.log10(bare_ploop/k**3)
    lnormed = plt.plot(lnx, lny,
                        # lnx[lnx>=min_logn], lny[lnx>=min_logn],
                       'k-.', label=f'Straight chain, b={b:0.1f}nm')
    # plt.plot(np.log10(bare_n), np.log10(wlc.sarah_looping(bare_n/2/wlc.default_lp)/(2*wlc.default_lp)**2))

    plt.xlim([min_logn*0.95, np.max(x)*1.05])
    if rmax_or_ldna == 'rmax':
        plt.ylim([-11, -6])
    elif rmax_or_ldna == 'ldna':
        plt.ylim([-13, -5])

    # # start at nearest
    # ticks = np.arange(np.ceil(min_logn), np.floor(np.max(x))+1)
    # plt.xticks

    if rmax_or_ldna == 'rmax':
        plt.xlabel('Total linker length (bp)')
    elif rmax_or_ldna == 'ldna':
        plt.xlabel('Genomic distance (bp)')
    plt.ylabel(r'$P_\mathrm{loop}\;\;\;(\mathrm{nm}^{-3})$')

    # plt.legend([fill, l100, lnormed], ['Average $\pm$ 95\%',
    #         'Straight chain, b=100nm', f'Straight chain, b={b:0.2f}nm'],
    plt.legend(loc='lower center')

    plt.savefig(f'plots/PRL/fig5_{named_sim}_{rmax_or_ldna}.pdf', bbox_inches='tight')




def load_looping_statistics_heterogenous_chains(*, dir=None, file_re=None, links_fmt=None, greens_fmt=None, named_sim=None):
    """Load in looping probabilities for all example chains of a given type
    done so far.

    Specify how to find the files via the directory dir, a regex that can
    extract the "num_nucs" and "chain_id" from the folder name, a format string that
    expands num_nucs, chain_id into the file name holding the linker lengths
    for that chain, and another format string that expands into the filename
    holding the greens function.

    OR: just pass named_sim='mu56' or 'links31-to-52' to load in exponential chains with
    mean linker length 56 or uniform linker chain with lengths from 31-52,
    resp.
    """
    if named_sim is not None:
        file_re = re.compile("([0-9]+)nucs_chain([0-9]+)")
        links_fmt = 'linker_lengths_{num_nucs}nucs_chain{chain_id}_{num_nucs}nucs.npy'
        greens_fmt = 'kinkedWLC_greens_{num_nucs}nucs_chain{chain_id}_{num_nucs}nucs.npy'
        if named_sim == 'mu56':
            #directory in which all chains are saved
            loops_dir = Path('csvs/Bprops/0unwraps/heterogenous/exp_mu56')
        elif named_sim == 'links31-to-52':
            loops_dir = Path('csvs/Bprops/0unwraps/heterogenous/links31to52')
        else:
            raise ValueError('Unknown sim type!')
        cache_csv = Path(loops_dir/f'looping_probs_heterochains_{named_sim}_0unwraps.csv')
        if cache_csv.exists():
            df = pd.read_csv(cache_csv)
            df = df.set_index(['num_nucs', 'chain_id']).sort_index()
            return df
    #Create one data frame per chain and add to this list; concatenate at end
    list_dfs = []
    #first load in chains of length 100 nucs
    for chain_folder in loops_dir.glob('*'):
        match = file_re.match(chain_folder.name)
        if match is None:
            continue
        num_nucs, chain_id = match.groups()
        try:
            links = np.load(chain_folder
                    /links_fmt.format(chain_id=chain_id, num_nucs=num_nucs))
            greens = np.load(chain_folder
                    /greens_fmt.format(chain_id=chain_id, num_nucs=num_nucs))
        except FileNotFoundError:
            print(f'Unable to find (num_nucs,chain_id)=({num_nucs},{chain_id}) in {chain_folder}')
            continue
        df = pd.DataFrame(columns=['num_nucs', 'chain_id', 'nuc_id', 'ldna', 'rmax', 'ploops'])
        #only including looping statistics for 2 nucleosomes onwards when plotting though
        df['ldna'] = convert.genomic_length_from_links_unwraps(links, unwraps=0)
        df['rmax'] = convert.Rmax_from_links_unwraps(links, unwraps=0)
        df['ploops'] = greens[0,:]
        df['num_nucs'] = int(num_nucs)
        df['chain_id'] = int(chain_id)
        df['nuc_id'] = np.arange(1, len(df)+1)
        list_dfs.append(df)
    #Concatenate list into one data frame
    df = pd.concat(list_dfs, ignore_index=True, sort=False)
    df = df.set_index(['num_nucs', 'chain_id']).sort_index()
    df.to_csv(cache_csv)
    return df

def plot_looping_probs_hetero_avg(df, **kwargs):
    #df2 = df.sort_values('ldna')
    fig, ax = plt.subplots(figsize=(default_width, default_height))
    #first just plot all chains
    palette = sns.cubehelix_palette(n_colors=len(df.groupby(['num_nucs', 'chain_id'])))
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
