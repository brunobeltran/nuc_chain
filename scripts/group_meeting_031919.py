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
params = {'axes.edgecolor': 'black', 'axes.grid': False, 'axes.facecolor': 'white', 'axes.titlesize': 20.0,
'axes.linewidth': 0.75, 'backend': 'pdf','axes.labelsize': 18,'legend.fontsize': 18,
'xtick.labelsize': 18,'ytick.labelsize': 18,'text.usetex': False,'figure.figsize': [7, 5],
'mathtext.fontset': 'stixsans', 'savefig.format': 'pdf', 'xtick.bottom':True, 'xtick.major.pad': 5, 'xtick.major.size': 5, 'xtick.major.width': 0.5,
'ytick.left':True, 'ytick.right':False, 'ytick.major.pad': 5, 'ytick.major.size': 5, 'ytick.major.width': 0.5, 'ytick.minor.right':False, 'lines.linewidth':2}

plt.rcParams.update(params)

#Mouse data
def plot_looping_within_contact_radius(a=10, lp=50, loglog=True):
    """Plot probability that 2 ends will form a loop within contact radius a, in nm,
    as a function of dimensionless chain length N=Rmax/2lp, where lp is in nm.
    Plots kinked model vs. bare WLC looping probabilities for both Rlinks and Llinks."""

    #convert a and lp to basepairs
    a_in_bp = a / ncg.dna_params['lpb']
    lp_in_bp = lp / ncg.dna_params['lpb']

    #load in linker lengths used to simulate nucleosome positions in mice (mu=45bp)
    links = np.load('csvs/Bprops/0unwraps/heterogenous/Sarah/mice2/linker_lengths_101nucs.npy')
    #Rlinks -- linkers right of TSS, starting with 60bp between -1 and +1 nuc
    Rlinks = links[50:]
    #Llinks -- linkers left of TSS, starting with 15bp between -2 and -1 nuc
    Llinks = links[0:50]
    #reverse so linker from -1 to -2 nuc is first
    Llinks_rev = Llinks[::-1]
    total_links = np.concatenate((Llinks_rev, Rlinks))

    #cumulative chain length including burried basepairs
    unwrap = 0
    #plot as positive distance from TSS in bp
    ldna_Rlinks = convert.genomic_length_from_links_unwraps(Rlinks, unwraps=unwrap) #max WLC chain length in bp
    #plot as negative distance from TSS in bp
    ldna_Llinks = -1*convert.genomic_length_from_links_unwraps(Llinks_rev, unwraps=unwrap) #max WLC chain length in bp

    #load in calculated looping probabilities
    loops = pd.read_csv('../../data_for_Sarah/mice2_looping_probs_a=10nm_102nucs.csv')

    #50th nucleosome or row 49, corresponds to -2nuc
    loops_to_plot = loops['49']
    #only interested in plotting looping with nucleation site (-2 nuc)
    Lloops = np.concatenate((loops_to_plot[0:49], [loops_to_plot[50]]))
    Lloops = Lloops[::-1]
    #linkers left of -2 nuc
    #ldna_leftnuc = ldna_Llinks[1:]

    #linkers right of -2 nuc
    Rloops = loops_to_plot[51:]
    #ldna_rightnuc = np.concatenate(([ldna_Llinks[0]], ldna_Rlinks))


    fig, ax = plt.subplots(figsize=(6.25, 4.89))
    colors = sns.color_palette("BrBG", 9)
    ax.plot(ldna_Rlinks, Rloops, '-o', lw=2, markersize=4, color=colors[1], label='Right of TSS')
    ax.plot(ldna_Llinks, Lloops, '-o', lw=2, markersize=4, color=colors[-2], label='Left of TSS')
    plt.xlabel(r'Distance from TSS (bp)')
    plt.ylabel(f'Looping probability, a={a}nm')
    plt.subplots_adjust(left=0.16, bottom=0.15, top=0.98, right=0.98)
    plt.yscale('log')
    plt.legend(loc=0)
    #plt.ylim([10**-4.5, 10**-1.8])
    plt.savefig(f'plots/lp{lp}nm_a{a}nm_left_right_contact_probability_mice2.png')


