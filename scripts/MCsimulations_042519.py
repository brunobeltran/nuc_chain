"""Script to analyze output of MC simulations of nucleosome chains."""
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

from nuc_chain import geometry as ncg
from nuc_chain import linkers as ncl
from MultiPoint import propagator
from nuc_chain import fluctuations as wlc
from nuc_chain.linkers import convert
from nuc_chain import visualization as vis
from mpl_toolkits.axes_grid1 import make_axes_locatable

#These follow Andy's plotting preferences for
params = {'axes.edgecolor': 'black', 'axes.grid': True, 'axes.titlesize': 20.0,
'axes.linewidth': 0.75, 'backend': 'pdf','axes.labelsize': 18,'legend.fontsize': 18,
'xtick.labelsize': 18,'ytick.labelsize': 18,'text.usetex': False,'figure.figsize': [7, 5],
'mathtext.fontset': 'stixsans', 'savefig.format': 'pdf', 'xtick.bottom':True, 'xtick.major.pad': 5, 'xtick.major.size': 5, 'xtick.major.width': 0.5,
'ytick.left':True, 'ytick.right':False, 'ytick.major.pad': 5, 'ytick.major.size': 5, 'ytick.major.width': 0.5, 'ytick.minor.right':False, 'lines.linewidth':2}

plt.rcParams.update(params)

teal_flucts = '#387780'
red_geom = '#E83151'
dull_purple = '#755F80'
rich_purple = '#e830e8'

dir_default='csvs/MC/41bp_notrans'

def extract_MC(dir=dir_default, timestep=55, v=1):
    """Extract the positions and orientation traids of each nucleosome in a MC simulation."""
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
    #^above files saved in csvs/MLC in npy format

def check_equilibration(maxtimestep=55):
    """Plot the end-to-end distance of the simulated structure vs. time to see if it's equilibrated"""
    MC_sqrtR2 = np.zeros((maxtimestep+1,))

    for i in range(maxtimestep):
        rots, pos = extract_MC(timestep=i, v=1)
        pos = pos - pos[0, :]
        r2 = np.array([pos[i, :].T@pos[i, :] for i in range(pos.shape[0])])
        MC_sqrtR2[i] = np.sqrt(r2[-1])

    fig, ax = plt.subplots()
    ax.plot(np.arange(0, maxtimestep+1), MC_sqrtR2)
    plt.xlabel('Time step')
    plt.ylabel('End-to-end distance (nm)')

    
def plot_R2_MC_vs_theory(links=np.tile(36, 6999), unwrap=0, num_sims=10, maxtimestep=55, plot_sims=False, simdir='csvs/MC/36bp_notrans'):
    """Plot the end-to-end distance predicted by the theory against the R^2 of the simulated structure."""
    link = links[0] #linker length (36bp)
    R2, Rmax, kuhn = wlc.R2_kinked_WLC_no_translation(links, plotfig=False, unwraps=unwrap)
    r2 = np.concatenate(([0], R2))
    rmax = np.concatenate(([0], Rmax))
    fig, ax = plt.subplots()

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
            plt.loglog(rmax, np.sqrt(MCr2), color=dull_purple, alpha=0.4, label=mylabel)
    
    plt.loglog(rmax, np.mean(MCr_allsim, axis=1), color=dull_purple, lw=3, label='Simulation')
    plt.loglog(rmax, np.sqrt(r2), color='k', lw=3, label='Theory')
    plt.legend()
    plt.xlabel(r'Total linker length (nm)')
    plt.ylabel(r'$\sqrt{\langle R^2 \rangle}$')
    plt.title('Homogeneous Chain 36bp linkers')
    plt.subplots_adjust(left=0.15, bottom=0.16, top=0.93, right=0.98)
    #plt.savefig('plots/end-to-end-distance-theory-vs-simulation_36bp_0unwraps.png')





