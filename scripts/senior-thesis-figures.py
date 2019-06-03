"""Script to generate figures for Beltran & Kannan et. al.

Two figures were made by hand. Figure 1 is a pair of blender renderings. The
relevant blend file names are simply mentioned below.

Where data has to be pre-computed, the procedure is mentioned."""
import re
from pathlib import Path
import pickle

import matplotlib.cm as cm
import matplotlib.ticker as tck
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import scipy
from scipy import stats
from scipy.optimize import curve_fit
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from nuc_chain import geometry as ncg
from nuc_chain import linkers as ncl
from nuc_chain import rotations as ncr
from MultiPoint import propagator
from nuc_chain import fluctuations as wlc
from nuc_chain import visualization as vis
from nuc_chain.linkers import convert

# Plotting parameters
# PRL Font preference: computer modern roman (cmr), medium weight (m), normal shape
cm_in_inch = 2.54
# column size is 8.6 cm
col_size = 10 / cm_in_inch
default_width = 1.0*col_size
aspect_ratio = 4/7
default_height = aspect_ratio*default_width
plot_params = {
    'backend': 'pdf',
    'savefig.format': 'pdf',
    'text.usetex': True,
    'font.size': 10,

    'figure.figsize': [default_width, default_height],
    'figure.facecolor': 'white',

    'axes.grid': False,
    'axes.edgecolor': 'black',
    'axes.facecolor': 'white',

    'axes.titlesize': 10.0,
    'axes.labelsize': 10,
    'legend.fontsize': 8.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'axes.linewidth': 1,

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
        vis.visualize_chain(entry_rots, entry_pos, linkers, unwraps=unwraps, plot_spheres=True, **kwargs)

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
        (in log units, to be consistent with draw_triangle)
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
    x0, y0 = [base**x for x in x0]
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

#link_ix, unwrap_ix, rise, angle, radius = ncg.tabulate_rise(dp_f=ncg.dp_omega_exit)

def plot_fig31_rise_vs_linker_length():
    fig, ax = plt.subplots(figsize=(1.2*default_width, default_height))
    links = np.arange(10, 101)
    #kuhns1to250 = np.load('csvs/kuhns_1to250links_0to146unwraps.npy')
    #calculate the 'phi' angle corresponding to twist due to linker
    phis_dp_omega_exit = np.zeros(links.size)
    for i, link in enumerate(links):
        dP, Onext = ncg.dp_omega_exit(link, unwrap=0)
        phi, theta, alpha = ncr.phi_theta_alpha_from_R(Onext)
        #record angles in units of pi
        phis_dp_omega_exit[i] = phi/np.pi + 1
    
    plt.plot(links, rise[0:91,0], linewidth=0.5)
    plt.scatter(links, rise[0:91,0], c=phis_dp_omega_exit, cmap='Spectral', s=3);
    plt.xlabel('Linker length (bp)')
    plt.ylabel(r'Rise (nm)')
    plt.subplots_adjust(left=0.1, bottom=0.19, top=0.95, right=0.97)
    cb = plt.colorbar(ticks=[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2])
    cb.set_label(r'$\phi$')
    cb.ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    #cb.ax.yaxis.set_yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2], 
    #                    [r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$',
    #                     r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$', r'$2\pi$'])
    fig.text(0.13, 0.47, r'38 bp', size=10)
    fig.text(0.12, 0.57, r'36 bp', size=10)
    plt.savefig('plots/thesis/fig3.1_rise-vs-linker-length.pdf')


default_lis = [36, 38]
default_colors = [red_geom, rich_purple]
def plot_fig32a(lis=default_lis, colors=None):
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

def plot_fig32b():
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

def render_fig32b_chains(**kwargs):
    for li in [36, 38, 41, 47]:
        render_chain(14*[li], **kwargs)

def extract_MC(dir, timestep=55, v=0):
    "Extract the positions and orientation traids of each nucleosome in a MC simulation."
    dir = Path(dir)
    if dir.is_dir() is False:
        raise ValueError(f'The director {dir} does not exist')
    entry_pos = np.loadtxt(dir/f'r{timestep}v{v}')
    entry_us = np.loadtxt(dir/f'u{timestep}v{v}')
    entry_t3 = entry_us[:, 0:3] #U
    entry_t1 = entry_us[:, 3:] #V
    entry_t2 = np.cross(entry_t3, entry_t1, axis=1) #UxV
    num_nucs = entry_pos.shape[0]
    entry_rots = []
    for i in range(num_nucs):
        #t1, t2, t3 as columns
        rot = np.eye(3)
        rot[:, 0] = entry_t1[i, :] #V
        rot[:, 1] = entry_t2[i, :] #UxV
        rot[:, 2] = entry_t3[i, :] #U
        entry_rots.append(rot)
    entry_rots = np.array(entry_rots)
    return entry_rots, entry_pos

def plot_fig33_R2_MC_vs_theory(links=np.tile(36, 6999), unwrap=0, num_sims=10, maxtimestep=55, plot_sims=True, simdir='csvs/MC/36bp_notrans'):
    """Plot the end-to-end distance predicted by the theory against the R^2 of the simulated structure."""
    link = links[0] #linker length (36bp)
    R2, Rmax, kuhn = wlc.R2_kinked_WLC_no_translation(links, plotfig=False, unwraps=unwrap)
    r2 = np.concatenate(([0], R2))
    rmax = np.concatenate(([0], Rmax))
    fig, ax = plt.subplots(figsize=(0.85*default_width, 1.1*default_height))

    MCr_allsim = np.ones((rmax.size, num_sims))
    #plot results from the 55th time step of 10 different simulations
    for i in range(1, num_sims):
        rots, pos = extract_MC(dir=simdir, timestep=maxtimestep, v=i)
        #shift MC positions so beginning of polymer coincides with the origin
        pos = pos - pos[0, :]
        #calculate end-to-end distance
        MCr2 = np.array([pos[i, :].T@pos[i, :] for i in range(pos.shape[0])])
        MCr_allsim[:, i-1] = np.sqrt(MCr2)
        #plot each of the individual simulated end-to-end distances
        if plot_sims:
            if i==1:
                mylabel = 'Simulation'
            else:
                mylabel = None
            plt.loglog(rmax, np.sqrt(MCr2), color=teal_flucts, lw=0.5, alpha=0.25, label=mylabel)
    
    plt.loglog(rmax, np.mean(MCr_allsim, axis=1), color=teal_flucts, lw=2, label='Simulation Average')
    plt.loglog(rmax, np.sqrt(r2), color='k', lw=2, label='Theory')
    plt.legend(loc=2)
    plt.xlabel(r'Total linker length (nm)')
    plt.ylabel(r'$\sqrt{\langle R^2 \rangle}$')
    plt.ylim([3, 2000])
    plt.title('Homogeneous Chain 36bp linkers')
    plt.subplots_adjust(left=0.17, bottom=0.19, top=0.91, right=0.98)
    plt.savefig('plots/thesis/theory-vs-simulation_36bp_0unwraps.pdf')

def plot_fig34_kuhns_multiple_unwraps():
    kuhns = np.load('csvs/kuhns_1to250links_0to146unwraps.npy')
    links = np.arange(1, 251)
    fig, ax = plt.subplots(figsize=(default_width, default_height))
    ax.plot(links, kuhns[:, 0], '-o', markersize=1, lw=0.75, color=teal_flucts, label='0 unwraps')
    ax.plot(links, kuhns[:, 21], '-o', markersize=1, lw=0.75, color=red_geom, label='21 unwraps')
    ax.plot(links, kuhns[:, 42], '-o', markersize=1, lw=0.75, color=dull_purple, label='42 unwraps')
    #half-unwrapped
    #ax.plot(links, kuhns[:, 147], '-o', markersize=1, lw=0.75, color=rich_purple, label='147 unwraps')
    plt.legend()
    plt.xlabel('Fixed linker length (bp)')
    plt.ylabel('Kuhn length (nm)')
    plt.subplots_adjust(left=0.12, bottom=0.20, top=0.98, right=0.98)
    plt.savefig('plots/thesis/fig34_kuhns-multiple-unwraps.pdf')

def render_fig34_chains(**kwargs):
    links = np.tile(38, 20)
    colors = [teal_flucts, red_geom, dull_purple]
    for i, unwrap in enumerate([0, 21, 42]):
        col = colors[i].lstrip('#') #string of the form #hex
        #convert hex color to RGB tuple of the form (0.0 <= floating point number <= 1.0, "", "")
        col = tuple(int(col[i:i+2], 16)/256 for i in (0, 2, 4))
        render_chain(links, unwraps=unwrap, nucleosome_color=col, **kwargs)

def plot_fig35_kuhns_long_linkers():
    links = np.load('csvs/linker_lengths_homogenous_so_far.npy')
    kuhns = np.load('csvs/kuhns_homogenous_so_far.npy')
    fig, ax = plt.subplots(figsize=(default_width, default_height))
    ax.plot(links, kuhns, '-o', markersize=1, lw=0.75, color=teal_flucts, label='Chromatin')
    ax.plot(np.linspace(min(links), max(links), 1000), np.tile(100, 1000), '--', label='Bare DNA', color=dull_purple)
    plt.xlabel('Fixed linker length (bp)')
    plt.ylabel('Kuhn length (nm)')
    plt.xscale('log')
    plt.legend(loc=4) #lower right
    plt.subplots_adjust(left=0.13, bottom=0.20, top=0.98, right=0.99)
    plt.tick_params(left=True, right=False, bottom=True, length=4)
    plt.savefig('plots/thesis/fig35_kuhns-long-linkers.pdf')

def plot_fig36(mu=41):
    """use scripts/r2-tabulation.py and wlc.aggregate_existing_kuhns to create
    the kuhns_so_far.csv file."""
    fig, ax = plt.subplots(figsize=(default_width, default_height))
    # index: variance_type, type, mu, variance, unwrap
    # columns: slope, intercept, rvalue, pvalue, stderr, b
    all_kuhns = pd.read_csv('./csvs/kuhns_so_far.csv', index_col=np.arange(5))
    kg = all_kuhns.loc['box', 'geometrical', mu].reset_index()
    kg = kg.sort_values('variance')
    ax.plot(kg['variance'].values, kg['b'].values, '--^', markersize=3, label='Zero-temperature',
            color=red_geom)
    kf = all_kuhns.loc['box', 'fluctuations', mu].reset_index()
    kf = kf.sort_values('variance')
    ax.plot(kf['variance'].values, kf['b'].values, '-o', markersize=3, label='Fluctuating',
            color=teal_flucts)
    rdf = pd.read_csv('./csvs/r2/r2-fluctuations-exponential-link-mu_41-0unwraps.csv')
    b = rdf['kuhn'].mean()
    xlim = plt.xlim()
    plt.plot([-10, 50], [b, b], 'k-.', label='Maximum Entropy')
    plt.xlim(xlim)
    ax.set_ylim([0, 100])
    plt.xlabel('Linker length variability $\pm\sigma$ (bp)')
    plt.ylabel('Kuhn length (nm)')
    plt.legend()
    fig.text(1.3, 0, r'$\pm 0 bp$', size=9)
    fig.text(1.6, 0, r'$\pm 2 bp$', size=9)
    fig.text(1.9, 0, r'$\pm 6 bp$', size=9)
    # plt.subplots_adjust(left=0.07, bottom=0.15, top=0.92, right=0.97)
    plt.tight_layout()
    plt.savefig('./plots/PRL/fig-3-kuhn_length_vs_window_size_41_sigma0to40.pdf',
               bbox_inches='tight')

def render_fig36_chains(mu=41, sigmas=[0, 2, 6]):
    for sigma in sigmas:
        sign_bit = 2*np.round(np.random.rand(N)) - 1
        render_chain(mu + sign_bit*np.random.randint(sigma+1), size=(N,))

def plot_fig37_r2_exponential(mu=36, colors=None):
    """The r2 of the 36bp exponential chain (0 unwrapping) compared to the
    wormlike chain with the corresponding Kuhn length."""
    fig, ax = plt.subplots(figsize=(default_width, 0.65*default_width))
    x = np.logspace(0, 7, 100)
    #compare to bare DNA WLC, default lp is 50nm
    y = np.sqrt(wlc.r2wlc(x))
    plt.plot(x, y, '.', color='k', markersize=1)
    #plot exponential chains
    rdf = pd.read_csv('./csvs/r2/r2-fluctuations-exponential-link-mu_36-0unwraps.csv')
    try:
        del rdf['Unnamed: 0']
    except:
        pass
    for i, chain in rdf.groupby(['mu', 'chain_id']):
        chain.iloc[0,0] = 1
        chain.iloc[0,1] = 1
        plt.plot(chain['rmax'], np.sqrt(chain['r2']), color=dull_purple, alpha=0.3, lw=0.5)
        break

    lp_bestfit = rdf['kuhn'].mean()/2
    y = np.sqrt(wlc.r2wlc(x, lp_bestfit))
    plt.plot(x, y, '-', color=teal_flucts)
    legend = [r'Bare DNA'] \
           + [r'$\langle L_i \rangle= 36bp$'] \
           + [r'WLC, $l_p \approx 15nm$']
    plt.legend(legend, loc=2)

    for i, chain in rdf.groupby(['mu', 'chain_id']):
        chain.iloc[0,0] = 1
        chain.iloc[0,1] = 1
        plt.plot(chain['rmax'], np.sqrt(chain['r2']), color=dull_purple, alpha=0.3, lw=0.5)
    plt.plot(x, y, '-', color=teal_flucts)

    plt.xlabel('Total linker length (nm)')
    plt.ylabel(r'$\sqrt{\langle R^2 \rangle}$')

    xmin = 1
    ymin = xmin
    ymax = 2000
    xmax = 4000
    # bands representing different regimes of the R^2
    plt.fill_between(x, ymin, ymax, where=x<12, color=[0.96, 0.95, 0.95])
    plt.fill_between(x, ymin, ymax, where=((x>=12)&(x<250)), color=[0.99, 0.99, 0.99])
    plt.fill_between(x, ymin, ymax, where=x>=250, color=[0.9, 0.9, 0.91])

    # power law triangle for the two extremal regimes
    corner = draw_power_law_triangle(1, [np.log10(2), np.log10(3)], 0.5, 'up')
    plt.text(3, 15, '$L^1$')
    corner = draw_power_law_triangle(1/2, [np.log10(450), np.log10(50)], 0.8, 'down')
    plt.text(900, 20, '$L^{1/2}$')

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Total linker length (nm)')
    plt.ylabel(r'End-to-end distance (nm)')
    plt.tight_layout()
    plt.savefig('plots/thesis/fig37_r2-exponential.pdf', bbox_inches='tight')

def plot_fig38():
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
    # plt.subplots_adjust(left=0.14, bottom=0.15, top=0.98, right=0.99)
    plt.xlabel(r'$\langle L_i \rangle$ (bp)')
    plt.ylabel(r'Kuhn length (nm)')
    plt.tight_layout()
    plt.savefig('plots/PRL/fig4b_kuhn_exponential.pdf', bbox_inches='tight')

def plot_fig39_homo_loop():
    kink41 = np.load(f'csvs/Bprops/0unwraps/41link/kinkedWLC_greens_41link_0unwraps_1000rvals_50nucs.npy')
    kink47 = np.load(f'csvs/Bprops/0unwraps/47link/kinkedWLC_greens_47link_0unwraps_1000rvals_50nucs.npy')
    bare41 = np.load(f'csvs/Bprops/0unwraps/41link/bareWLC_greens_41link_0unwraps_1000rvals_50nucs.npy')
    integrals = [kink47, kink41, bare41]
    labels = ['47bp', '41bp', 'Straight chain']
    links_list = [np.tile(47, 50), np.tile(41, 50), np.tile(41, 50)]
    plot_prob_loop_vs_fragment_length(integrals, labels, links_list, unwrap=0, nucmin=2)
    plt.subplots_adjust(left=0.17, bottom=0.20, top=0.96, right=0.97)
    plt.savefig('plots/thesis/fig39_looping-homo.pdf')

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

    fig, ax = plt.subplots(figsize=(default_width, 1.1*default_height))
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
    for i in range(len(labels)):
        ldna = convert.genomic_length_from_links_unwraps(links[i], unwraps=unwrap)
        ploops = integrals[i][0, indmin:]
        pldna = ldna[inds]
        ax.loglog(pldna, ploops, '-o', markersize=2, linewidth=1,
            color=colors[i], label=labels[i], **kwargs)
    ax.legend(loc=(0.32, 0.03), frameon=False, fontsize=10)
    plt.xlabel('Genomic distance (bp)')
    plt.ylabel(r'$P_\mathrm{loop}\;\;\;(\mathrm{bp}^{-3})$')

def render_fig39_chains(**kwargs):
    color_red = sns.color_palette("hls", 8)[0]
    colors = [color_red, '#D9A725', '#387780']
    for i, link in enumerate([47, 41, 41]):
        col = colors[i].lstrip('#') #string of the form #hex
        #convert hex color to RGB tuple of the form (0.0 <= floating point number <= 1.0, "", "")
        col = tuple(int(col[i:i+2], 16)/256 for i in (0, 2, 4))
        links = np.tile(link, 10)
        render_chain(links, unwraps=0, nucleosome_color=col, **kwargs)

def plot_fig310(df=None, rmax_or_ldna='rmax', named_sim='mu56'):
    fig, ax = plt.subplots(figsize=(default_width, 1.15*default_height))
    n = rmax_or_ldna
    # first set sim-specific parameters, draw scaling triangles at manually
    # chosen locations
    if (named_sim, rmax_or_ldna) == ('mu56', 'ldna'):
        draw_power_law_triangle(-3/2, x0=[3.8, -7.1], width=0.4, orientation='up')
        plt.text(10**(3.95), 10**(-6.8), '$L^{-3/2}$')
        # manually set thresholds to account for numerical instability at low n
        min_n = 10**2.6
    elif (named_sim, rmax_or_ldna) == ('mu56', 'rmax'):
        draw_power_law_triangle(-3/2, x0=[3.0, -7.5], width=0.4, orientation='up')
        plt.text(10**3.1, 10**(-7.3), '$L^{-3/2}$')
        min_n = 10**2.2
    elif (named_sim, rmax_or_ldna) == ('links31-to-52', 'rmax'):
        draw_power_law_triangle(-3/2, x0=[3.0, -7.5], width=0.4, orientation='up')
        plt.text(10**3.1, 10**(-7.3), '$L^{-3/2}$')
        min_n = 10**2.0
    elif (named_sim, rmax_or_ldna) == ('links31-to-52', 'ldna'):
        draw_power_law_triangle(-3/2, x0=[3.5, -7], width=0.4, orientation='up')
        plt.text(10**3.6, 10**(-6.8), '$L^{-3/2}$')
        min_n = 10**2.5
    if df is None:
        df = load_looping_statistics_heterogenous_chains(named_sim=named_sim)
    # if the first step is super short, we are numerically unstable
    df.loc[df['rmax'] <= 5, 'ploops'] = np.nan
    # if the output is obviously bad numerics, ignore it
    df.loc[df['ploops'] > 10**(-4), 'ploops'] = np.nan
    df.loc[df['ploops'] < 10**(-13), 'ploops'] = np.nan
    df = df.dropna()
    df = df.sort_values(n)
    # # # rolling window doesn't seem to work
    # # get an estimator of the variance from a rolling window
    # # window size chosen by eye
    # df['t'] = pd.to_datetime(np.log10(df[n])*60*10**9) # to minutes
    # dft = df.set_index('t')
    # rolled = dft.rolling('10s').apply(np.nanmean, raw=False)
    # rolled_ste = dft.rolling('10s').apply(lambda df: np.nanstd(df)/np.sqrt(len(df)), raw=False)
    # # rolled = df.rolling(200).apply(np.nanmean, raw=False)
    # # rolled_ste = df.rolling(200).apply(lambda df: np.nanstd(df)/np.sqrt(len(df)), raw=False)
    # xgrid = rolled[n].values
    # y_pred = rolled['ploops'].values[xgrid > min_n]
    # sig = rolled_ste['ploops'].values[xgrid > min_n]

    # # # can't seem to get teh gaussian process fitting to work unless I take to
    # # # log space... something likely about the parameters or the kernel I'm
    # # # choosing?
    # # # should we just use rolling average?
    # # tricky! doing fitting in log space for x variable helps numerics
    # x = np.atleast_2d(df[logn].values.copy()).T
    # y = df['ploops'].values.copy().ravel()
    # sigma = np.interp(x, rolled[logn].values,
    #         rolled_std['ploops'].values,
    #         left=rolled_std['ploops'].values[0],
    #         right=rolled_std['ploops'].values[-1])
    # sigma = sigma.ravel()
    # # pandas rolling std doesn't like left boundary, but we can just fill in
    # # soemthing reasonable
    # sigma[np.isnan(sigma)] = np.nanmax(sigma)
    # # sigma += 10**-7 # to prevent numerical issues
    # # now fit to a gaussian process
    # if Path(f'csvs/gp_{named_sim}_{rmax_or_ldna}.pkl').exists():
    #     gp = pickle.load(open(f'csvs/gp_{named_sim}_{rmax_or_ldna}.pkl', 'rb'))
    # else:
    #     kernel = C(10.0, (1e-3, 1e3)) * RBF(0.5, (1e-1, 1))
    #     gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma**2, n_restarts_optimizer=10)
    #     gp.fit(x, y)
    #     pickle.dump(gp, open(f'gp_{named_sim}_{rmax_or_ldna}.pkl', 'wb'))
    # # recall fit done in log space
    # xgrid = np.linspace(max(np.log10(min_n), np.min(x)), np.max(x), 100).reshape(-1, 1)
    # y_pred, sig = gp.predict(xgrid, return_std=True)
    df_int = df.groupby(['num_nucs', 'chain_id']).apply(interpolated_ploop,
            rmax_or_ldna=rmax_or_ldna, n=np.logspace(np.log10(min_n), np.log10(df[n].max()), 1000))
    df_int_ave = df_int.groupby(n+'_interp')['ploops_interp'].agg(['mean', 'std', 'count'])
    df_int_ave = df_int_ave.reset_index()
    xgrid = df_int_ave[n+'_interp'].values
    y_pred = df_int_ave['mean'].values
    sig = df_int_ave['std'].values/np.sqrt(df_int_ave['count'].values - 1)
    # 95% joint-confidence intervals, bonferroni corrected
    ste_to_conf = scipy.stats.norm.ppf(1 - (0.05/1000)/2)
    # plot all the individual chains, randomly chop some down to make plot look
    # nicer
    palette = sns.cubehelix_palette(n_colors=len(df.groupby(['num_nucs', 'chain_id'])))
    ord = np.random.permutation(len(palette))
    for i, (label, chain) in enumerate(df.groupby(['num_nucs', 'chain_id'])):
        num_nucs = int(label[0])
        max_nuc_to_plot = num_nucs*(1 - 0.2*np.random.rand())
        chain = chain[chain['nuc_id'] <= max_nuc_to_plot]
        chain = chain[chain[n] >= min_n]
        plt.plot(chain[n].values, chain['ploops'].values,
                 c=palette[ord[i]], alpha=0.15, lw=0.5, label=None)
    # bold a couple of the chains
    bold_c = palette[int(9*len(palette)/10)]
    if named_sim == 'mu56':
        chains_to_bold = [(100,1), (50,120), (100,112)]
    elif named_sim == 'links31-to-52':
        chains_to_bold = [(50, 1), (50, 3), (50, 5)]
        min_n = 10**2.7
    for chain_id in chains_to_bold:
        chain = df.loc[chain_id]
        chain = chain[chain[n] >= min_n]
        plt.plot(chain[n].values, chain['ploops'].values, c=bold_c, alpha=0.6,
                 label=None)
    fill = plt.fill_between(xgrid,
            y_pred - ste_to_conf*sig,
            y_pred + ste_to_conf*sig,
            alpha=.10, color='r')

    plt.plot(xgrid, y_pred, 'r-', label='Average $\pm$ 95\%')

    # load in the straight chain, in [bp] (b = 100nm/ncg.dna_params['lpb'])
    bare_n, bare_ploop = wlc.load_WLC_looping()
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
    x, y = bare_n*k, bare_ploop/k**3,
    lnormed = plt.plot(x[x >= min_n], y[x >= min_n],
                       'k-.', label=f'Straight chain, b={b:0.1f}nm')
    # also plot just the bare WLC
    b = 2*wlc.default_lp
    l100 = plt.plot(bare_n[bare_n>=min_n], bare_ploop[bare_n>=min_n], '-.', c=teal_flucts,
             label=f'Straight chain, b=100nm')
    # plt.plot(bare_n, wlc.sarah_looping(bare_n/2/wlc.default_lp)/(2*wlc.default_lp)**2)

    plt.xlim([10**(np.log10(min_n)*1), 10**(np.log10(np.max(df[n]))*0.99)])
    if rmax_or_ldna == 'rmax':
        plt.ylim([10**(-11), 10**(-6)])
    elif rmax_or_ldna == 'ldna':
        plt.ylim([10**(-13), 10**(-5)])
    plt.tick_params(axis='y', which='minor', left=False)

    if rmax_or_ldna == 'rmax':
        plt.xlabel('Total linker length (bp)')
    elif rmax_or_ldna == 'ldna':
        plt.xlabel('Genomic distance (bp)')
    plt.ylabel(r'$P_\mathrm{loop}\;\;\;(\mathrm{bp}^{-3})$')

    # plt.legend([fill, l100, lnormed], ['Average $\pm$ 95\%',
    #         'Straight chain, b=100nm', f'Straight chain, b={b:0.2f}nm'],
    plt.legend(loc='upper right', bbox_to_anchor=[0.9, 0.4])
    plt.yscale('log')
    plt.xscale('log')

    plt.tight_layout()
    plt.savefig(f'plots/thesis/fig310_{named_sim}_{rmax_or_ldna}.pdf', bbox_inches='tight')

def interpolated_ploop(df, rmax_or_ldna='ldna', n=np.logspace(2, 5, 1000),
                       ploop_col='ploops'):
    """Function to apply to the looping probabilities of a given chain to
    resample it to a fixed set of values."""
    n_col = rmax_or_ldna
    n = n[(n >= df[n_col].min()) & (n <= df[n_col].max())]
    ploop = np.interp(n, df[n_col].values, df[ploop_col].values,
            left=df[ploop_col].values[0], right=df[ploop_col].values[-1])
    return pd.DataFrame(np.stack([n, ploop]).T, columns=[n_col+'_interp', ploop_col+'_interp'])

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

def plot_fig43a_looping_TSS(a=10, loglog=True):
    """Plot probability that 2 ends will form a loop within contact radius a, in nm,
    as a function of dimensionless chain length N=Rmax/2lp, where lp is in nm.
    Plots kinked model vs. bare WLC looping probabilities for both Rlinks and Llinks."""

    fig, ax = plt.subplots(figsize=(default_width, default_height))
    #convert a to basepairs
    a_in_bp = a / ncg.dna_params['lpb']
    links35 = np.tile(35, 44)
    Rlinks = np.array([47, 21, 18, 15, 20, 17])
    Llinks = np.array([245, 30, 26, 15, 23, 35])
    #links to right of methylation site (50 in total)
    Rlinks = np.concatenate((Rlinks, links35))
    #links to left of methylation site (50 in total)
    Llinks = np.concatenate((Llinks, links35))
    #cumulative chain length including burried basepairs
    unwrap = 0
    #plot as positive distance from TSS in bp
    ldna_Rlinks = convert.genomic_length_from_links_unwraps(Rlinks, unwraps=unwrap) #max WLC chain length in bp
    #plot as negative distance from TSS in bp
    ldna_Llinks = -1*convert.genomic_length_from_links_unwraps(Llinks, unwraps=unwrap) #max WLC chain length in bp
    rvals = np.linspace(0.0, 1.0, 1000)
    integral_R = np.load('../deepti/csvs/Bprops/0unwraps/heterogenous/Sarah/Rlinks_1to50nucs/kinkedWLC_greens_Rlinks_50nucs_1000rvals.npy')
    integral_L = np.load('../deepti/csvs/Bprops/0unwraps/heterogenous/Sarah/Llinks_1to50nucs/kinkedWLC_greens_Llinks_50nucs_1000rvals.npy')
    Prob_a_Rlinks_kinked = wlc.prob_R_in_radius_a_given_L(a_in_bp, integral_R, rvals, Rlinks, unwrap)
    Prob_a_Llinks_kinked = wlc.prob_R_in_radius_a_given_L(a_in_bp, integral_L, rvals, Llinks, unwrap)
    #plot in units of nm^(-3)
    ax.plot(ldna_Rlinks, Prob_a_Rlinks_kinked/(ncg.dna_params['lpb']**3), '-o', markersize=3, color=teal_flucts)
    ax.plot(ldna_Llinks, Prob_a_Llinks_kinked/(ncg.dna_params['lpb']**3), '-o', markersize=3, color=teal_flucts)
    plt.xlabel(r'Distance from TSS (bp)')
    plt.ylabel(r'$P_\mathrm{loop}(a=10\mathrm{nm}; L)\;\;\;(\mathrm{nm}^{-3})$')
    plt.subplots_adjust(left=0.16, bottom=0.19, top=0.98, right=0.96)
    plt.yscale('log')
    plt.ylim([10**-3, 10**-0.4])
    plt.xlim([-10000, 10000])
    plt.savefig(f'plots/thesis/fig43a_looping-TSS-yeast.pdf', bbox_inches='tight')

def plot_fig43b_spreading_yeast():
    """Plot the relative enrichment of methylation spreading for the nucleosome positions
    extracted from the yeast TSS data."""
    fig, ax = plt.subplots(figsize=(default_width, default_height))
    y = np.loadtxt('csvs/ratio_lp50a10ad5hp0.0067379_yeast.txt')
    links35 = np.tile(35, 44)
    Rlinks = np.array([47, 21, 18, 15, 20, 17])
    Llinks = np.array([245, 30, 26, 15, 23, 35])
    #links to right of methylation site (50 in total)
    Rlinks = np.concatenate((Rlinks, links35))
    #links to left of methylation site (50 in total)
    Llinks = np.concatenate((Llinks, links35))
    #cumulative chain length including burried basepairs
    unwrap = 0
    #plot as positive distance from TSS in bp
    ldna_Rlinks = convert.genomic_length_from_links_unwraps(Rlinks, unwraps=unwrap) #max WLC chain length in bp
    #plot as negative distance from TSS in bp
    ldna_Llinks = -1*convert.genomic_length_from_links_unwraps(Llinks, unwraps=unwrap) #max WLC chain length in bp
    x = np.concatenate((ldna_Llinks[::-1], ldna_Rlinks))
    ax.plot(x, y, color='k')
    ax.set_xlabel(r'Distance from TSS (bp)')
    ax.set_ylabel('Relative enrichment')
    #plot inset using Crabtree data
    axins = inset_axes(ax, width="40%", height="40%", 
                    bbox_to_anchor=(.1, .1, .8, .8),
                    bbox_transform=ax.transAxes, loc=2)
    xcrabtree = np.array([-10256, -3077, -2241, -1485, -739, -309, -169, 489, 1746, 3087, 4400, 5300])
    REday0 = np.array([0.27, 0.13, 0.46, 0.12, 0.17, 0.33, 0.33, 0.31, 0.32, 0.27, 0.21, 0.33])
    REday5 = np.array([0.19, 0.40, 0.89, 1.55, 0.97, 1.25, 2.25, 3.57, 3.03, 2.09, 1.12, 0.14])
    ycrabtree = REday5/np.mean(REday0)
    axins.plot(xcrabtree, ycrabtree)
    axins.set_xlim([-10000, 10000])
    ax.set_xlim([-10000, 10000])
    #axins.set_ylabel('Relative enrichment', fontsize=8)
    #axins.set_xlabel('Distance from TSS (bp)', fontsize=8)
    plt.subplots_adjust(left=0.16, bottom=0.19, top=0.98, right=0.96)
    plt.savefig(f'plots/thesis/fig43b_spreading-TSS-yeast.pdf', bbox_inches='tight')

def plot_fig44a_wlc_inhomo_test7():
    """Plot looping probability with nuc site for WLC with two persistence lengths. 
    Parameters used for calculation are saved in loopdata_test7.mat"""

    fig, ax = plt.subplots(figsize=(default_width, default_height))
    #paramters from calculation, all in nm
    a = 5 #looping radius
    DEL = 15.3 #spacing between nucleosomes
    MINUS1NUC = -83.3 #distance between -1 nuc and TSS
    PLUS1NUC = 0
    NNUC = 50 #number of nucleosomes on each side of TSS
    LPB = 0.34 #nm per basepair (as used in calculation)

    Llinks = np.concatenate(([245], np.tile(45, 49)))
    Rlinks = np.tile(45, 49)
    #plot as negative distance from TSS in bp
    unwrap=0
    ldna_Llinks = -1*convert.genomic_length_from_links_unwraps(Llinks, unwraps=unwrap)
    ldna_Rlinks = convert.genomic_length_from_links_unwraps(Rlinks, unwraps=unwrap)
    x = np.concatenate((ldna_Llinks[::-1], [0], ldna_Rlinks))
    y = np.loadtxt('csvs/Sarah/pnucsite_test7.txt')
    ax.plot(x, y, '-o', markersize=3, color=teal_flucts)
    plt.xlabel(r'Distance from TSS (bp)')
    plt.ylabel(r'$P_\mathrm{loop}(a=5\mathrm{nm}; L)\;\;\;(\mathrm{nm}^{-3})$')
    plt.subplots_adjust(left=0.16, bottom=0.19, top=0.98, right=0.96)
    plt.yscale('log')
    #plt.ylim([10**-3, 10**-0.4])
    plt.xlim([-10000, 10000])
    plt.savefig(f'plots/thesis/fig44a_wlc-inhomo-looping.pdf', bbox_inches='tight')

def plot_fig44b_wlc_inhomo_spreading_test7():
    "Plot relative enrichment of methylation spreading using looping probabilities from test 7."""

    fig, ax = plt.subplots(figsize=(default_width, default_height))
    Llinks = np.concatenate(([245], np.tile(45, 49)))
    Rlinks = np.tile(45, 49)
    #plot as negative distance from TSS in bp
    unwrap=0
    ldna_Llinks = -1*convert.genomic_length_from_links_unwraps(Llinks, unwraps=unwrap)
    ldna_Rlinks = convert.genomic_length_from_links_unwraps(Rlinks, unwraps=unwrap)
    x = np.concatenate((ldna_Llinks[::-1], [0], ldna_Rlinks))
    y = np.loadtxt('csvs/Sarah/ratio_lp17.68a5ad5hp0.0067379_test7.txt')
    ax.plot(x, y, color='k')
    ax.set_xlabel(r'Distance from TSS (bp)')
    ax.set_ylabel('Relative enrichment')
    #plot inset using Crabtree data
    axins = inset_axes(ax, width="40%", height="40%", 
                    bbox_to_anchor=(.1, .1, .8, .8),
                    bbox_transform=ax.transAxes, loc=2)
    xcrabtree = np.array([-10256, -3077, -2241, -1485, -739, -309, -169, 489, 1746, 3087, 4400, 5300])
    REday0 = np.array([0.27, 0.13, 0.46, 0.12, 0.17, 0.33, 0.33, 0.31, 0.32, 0.27, 0.21, 0.33])
    REday5 = np.array([0.19, 0.40, 0.89, 1.55, 0.97, 1.25, 2.25, 3.57, 3.03, 2.09, 1.12, 0.14])
    ycrabtree = REday5/np.mean(REday0)
    axins.plot(xcrabtree, ycrabtree)
    axins.set_xlim([-10000, 10000])
    ax.set_xlim([-10000, 10000])
    plt.subplots_adjust(left=0.16, bottom=0.19, top=0.98, right=0.96)
    plt.savefig(f'plots/thesis/fig44b_wlc-inhomo-spreading.pdf', bbox_inches='tight')


