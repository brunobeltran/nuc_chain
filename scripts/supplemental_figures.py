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
from MultiPoint import propagator
from nuc_chain import fluctuations as wlc
from nuc_chain.linkers import convert
from mpl_toolkits.axes_grid1 import make_axes_locatable

#These follow Andy's plotting preferences for
params = {'axes.edgecolor': 'black', 'axes.grid': True, 'axes.titlesize': 20.0,
'axes.linewidth': 0.75, 'backend': 'pdf','axes.labelsize': 18,'legend.fontsize': 18,
'xtick.labelsize': 18,'ytick.labelsize': 18,'text.usetex': False,'figure.figsize': [7, 5], 
'mathtext.fontset': 'stixsans', 'savefig.format': 'pdf', 'xtick.bottom':True, 'xtick.major.pad': 5, 'xtick.major.size': 5, 'xtick.major.width': 0.5,
'ytick.left':True, 'ytick.right':False, 'ytick.major.pad': 5, 'ytick.major.size': 5, 'ytick.major.width': 0.5, 'ytick.minor.right':False, 'lines.linewidth':2}

plt.rcParams.update(params)

Klin = np.linspace(0, 10**5, 20000)
Klog = np.logspace(-3, 5, 10000)
Ks = np.unique(np.concatenate((Klin, Klog)))
#Kprops = wlc.tabulate_bareWLC_propagators(Ks)

#compare linker propagators generated by ODE solver vs. Quinn/Shafiq's partial fractions solution in Laplace space
def calculate_GKN_Quinn_vs_Deepti(Ns=np.array([0.1, 1.0, 10, 100]), l, l0, j, Ks=Ks):
    """Plotting code for G(K;N) vs K for a given l, l0, j. Plots one curve for each N."""

    Gs = np.zeros((Ns.size, Ks.size), 'complex')
    qGs = np.zeros((Ns.size, Ks.size), 'complex')
    for nn in range(Ns.size):
        for kk in range(Ks.size):
            qGs[nn, kk] = Kprops[kk].get_G(Ns[nn], l0, l)
            Gs[nn, kk] = wlc.get_G(Ks[kk], Ns[nn], l, l0, j)
        ax.loglog(Ks, np.abs(Gs[nn, :].real), color = colors[nn], label='ODE solution')
        ax.loglog(Ks, np.abs(qGs[nn, :].real), '--', color = colors[nn], label='Laplace solution')
        ax.loglog(Ks, np.sin(Ks*Ns[nn])/(Ks*Ns[nn]), ':', color = colors[nn], label='sin(KN)/(KN)')

    ax.legend(frameon=True)
    plt.xlabel('K')
    plt.ylabel('G(K;N)')
    plt.title(f'$l={l}, l_0={l0}, j={j}$')
    plt.legend([f'N={N}' for N in Ns])
    plt.ylim([10**-12, 2])
    plt.show()
    return Gs

def plot_GKN_Quinn_vs_Deepti(Ns=np.array([0.1, 1.0, 10, 100]), Ks=Ks):
    fig, ax = plt.subplots(figsize=(7.21, 5.19))
    colors = sns.color_palette()
    for nn in range(Ns.size):
        ax.loglog(Ks, Gs[nn, :].real, color = colors[nn], label=f'N={Ns[nn]}, ODE')
        ax.loglog(Ks, qGs[nn, :].real, '--', color = colors[nn], label=f'N={Ns[nn]}, Laplace')

    plt.legend(fontsize=14)
    plt.xlabel('K')
    plt.ylabel('G(K;N)')
    plt.title(f'$l={l}, l_0={l0}, j={j}$')
    plt.ylim([10**-8, 2])
    plt.xlim([10**-3, 10**4])
    plt.subplots_adjust(left=0.15, bottom=0.16, top=0.91, right=0.95)
    plt.savefig('plots/props/GKN_quinnWLC_vs_deeptiWLC_linker_prop.png')

def plot_GKN_rigid_rod_vs_kinked(Ns=np.array([0.1, 1.0, 10]), Ks=Ks):
    """Plotting code for G(K;N) vs K for a given l, l0, j. Plots one curve for each N."""
    fig, ax = plt.subplots(figsize=(7.21, 5.19))
    colors = sns.color_palette()
    for nn in range(3):
        ax.loglog(Ks, Gs[nn, :].real, color = colors[nn], label=f'N={Ns[nn]}, ODE solution')
        ax.loglog(Ks, np.sin(Ks*Ns[nn])/(Ks*Ns[nn]), ':', color = colors[nn], label=f'N={Ns[nn]}, sin(KN)/(KN)')

    plt.legend(fontsize=14)
    plt.xlabel('K')
    plt.ylabel('G(K;N)')
    plt.title(f'$l={l}, l_0={l0}, j={j}$')
    plt.ylim([10**-8, 2])
    plt.xlim([10**-3, 10**4])
    plt.subplots_adjust(left=0.15, bottom=0.16, top=0.91, right=0.95)
    plt.savefig('plots/props/GKN_rigid_rod_vs_deeptiWLC_linker_prop.png')


#plot Green's function for kinked vs. bare WLC
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






