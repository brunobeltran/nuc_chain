r"""
Deepti Kannan
08-23-18

Analysis of looping probabilities for Sarah's heterogenous chain:
  35 . . . 35, 23, 15, 26, 30, 245, nucleation site, 47, 21, 18, 15, 20, 17, 35 . . . 35
Nucleosome numbers:
50 . . . 6,   5,  4,  3,  2,  1,          0        ,    1,  2,  3,  4,  5,  6,  7, . . . 50
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from MultiPoint import propagator
from nuc_chain import fluctuations as wlc
from nuc_chain import geometry as ncg
from pathlib import Path
from scipy import stats
import seaborn as sns

params = {'axes.edgecolor': 'black', 'axes.facecolor': 'white', 'axes.grid': False, 
'axes.linewidth': 0.75, 'backend': 'pdf','axes.labelsize': 12,'legend.fontsize': 10,
'xtick.labelsize': 10,'ytick.labelsize': 10,'text.usetex': False,'figure.figsize': [5.75, 4.5], 
'mathtext.fontset': 'stixsans', 'savefig.format': 'pdf', 'xtick.major.pad': 4, 'xtick.major.size': 5, 'xtick.major.width': 0.5,
'ytick.major.pad': 4, 'ytick.major.size': 5, 'ytick.major.width': 0.5,}

plt.rcParams.update(params)
from multiprocessing import Pool
from functools import partial

"""All variables needed for analysis"""
Klin = np.linspace(0, 10**5, 20000)
Klog = np.logspace(-3, 5, 10000)
Kvals = np.unique(np.concatenate((Klin, Klog)))
#convert to little k -- units of inverse bp (this results in kmax = 332)
kvals = Kvals / (2*wlc.default_lp)
links35 = np.tile(35, 44)
Rlinks = np.array([47, 21, 18, 15, 20, 17])
Llinks = np.array([245, 30, 26, 15, 23, 35])
Rlinks_rev = Rlinks[::-1]
Llinks_rev = Llinks[::-1]

#links to right of methylation site (50 in total)
Rlinks = np.concatenate((Rlinks, links35))
#links to left of methylation site (50 in total)
Llinks = np.concatenate((Llinks, links35))

#cumulative chain length (linkers only) in bp
unwrap = 0
Rmax_Rlinks = wlc.Rmax_from_links_unwraps(Rlinks, unwraps=unwrap) #max WLC chain length in bp
Rmax_Llinks = wlc.Rmax_from_links_unwraps(Llinks, unwraps=unwrap) #max WLC chain length in bp

#Bare WLC green's functions (1000 rvals, 50 nucs) for lp = 50nm
# qintegral_R_lp50 = np.load('csvs/Bprops/0unwraps/Sarah/bareWLC_greens_Rlinks_50nucs.npy')
# qintegral_L_lp50 = np.load('csvs/Bprops/0unwraps/Sarah/bareWLC_greens_Llinks_50nucs.npy')
# #Bare WLC green's functions (1000 rvals, 50 nucs) for lp = 10.18nm
# qintegral_R_lp10 = np.load('csvs/Bprops/0unwraps/Sarah/bareWLC_greens_Rlinks_50nucs_lp10.npy')
# qintegral_L_lp10 = np.load('csvs/Bprops/0unwraps/Sarah/bareWLC_greens_Llinks_50nucs_lp10.npy')
# #Kinked WLC green's function (1000 rvals, 50 nucs)
# integral_R = np.load('csvs/Bprops/0unwraps/Sarah/kinkedWLC_greens_Rlinks_50nucs_1000rvals.npy')
# integral_L = np.load('csvs/Bprops/0unwraps/Sarah/kinkedWLC_greens_Llinks_50nucs_1000rvals.npy')

def bareWLC_greens(lp, props=None):
	"""Calculate greens functions for different bare WLC persistance lengths.
	Ran this function with lpvals = [20, 30, 40, 60, 70, 80]
	"""
	if props is None:
		props = wlc.tabulate_bareWLC_propagators(Kvals)

	lp_in_bp = lp / ncg.dna_params['lpb']
	Rfile = Path(f'csvs/Bprops/0unwraps/Sarah/bareWLC_greens_Rlinks_50nucs_lp{lp:.0f}.npy')
	if (Rfile.is_file() is False):
		#bare WLC propagators
		qprop_R = wlc.bareWLC_gprop(kvals, Rlinks, unwrap, props=props, lp=lp_in_bp)
		#bare WLC greens
		qintegral_R = wlc.BRN_fourier_integrand_splines(kvals, Rlinks, unwrap, Bprop=qprop_R)
		np.save(Rfile, qintegral_R)
	Lfile = Path(f'csvs/Bprops/0unwraps/Sarah/bareWLC_greens_Llinks_50nucs_lp{lp:.0f}.npy')
	if (Lfile.is_file() is False):
		#bare WLC propagators
		qprop_L = wlc.bareWLC_gprop(kvals, Llinks, unwrap, props=props, lp=lp_in_bp)
		#bare WLC greens
		qintegral_L = wlc.BRN_fourier_integrand_splines(kvals, Llinks, unwrap, Bprop=qprop_L)
		np.save(Lfile, qintegral_L)

	print(f'Saved G(R;L) for bare WLC with lp={lp}nm')

def plot_looping_kinked_vs_bare(a, lp, loglog=True):
	"""Plot probability that 2 ends will form a loop within contact radius a, in nm,
	as a function of dimensionless chain length N=Rmax/2lp, where lp is in nm. 
	Plots kinked model vs. bare WLC looping probabilities for both Rlinks and Llinks."""

	#convert a and lp to basepairs
	a_in_bp = a / ncg.dna_params['lpb']
	lp_in_bp = lp / ncg.dna_params['lpb']

	#dimensionless chain length N=Rmax/2lp
	Ns_Rlinks = Rmax_Rlinks / (2*lp_in_bp)
	Ns_Llinks = Rmax_Llinks / (2*lp_in_bp)

	rvals = np.linspace(0.0, 1.0, 1000)
	Prob_a_Rlinks_kinked = wlc.prob_R_in_radius_a_given_L(a_in_bp, integral_R, rvals, Rlinks, unwrap)
	Prob_a_Llinks_kinked = wlc.prob_R_in_radius_a_given_L(a_in_bp, integral_L, rvals, Llinks, unwrap)

	#load in appropriate integrals for bareWLC comparison
	Rfile = Path(f'csvs/Bprops/0unwraps/Sarah/bareWLC_greens_Rlinks_50nucs_lp{lp:.0f}.npy')
	if Rfile.is_file():
		qintegral_R = np.load(Rfile)
	else:
		raise ValueError(f'bareWLC greens function file does not exist for lp={lp}nm, Rlinks')
	
	Lfile = Path(f'csvs/Bprops/0unwraps/Sarah/bareWLC_greens_Llinks_50nucs_lp{lp:.0f}.npy')
	if Lfile.is_file():
		qintegral_L = np.load(Lfile)
	else:
		raise ValueError(f'bareWLC greens function file does not exist for lp={lp}nm, Llinks')	

	Prob_a_Rlinks_bare = wlc.prob_R_in_radius_a_given_L(a_in_bp, qintegral_R, rvals, Rlinks, unwrap)
	Prob_a_Llinks_bare = wlc.prob_R_in_radius_a_given_L(a_in_bp, qintegral_L, rvals, Llinks, unwrap)

	plt.rcParams.update({'figure.figsize': [5.75, 4.5]})
	fig, ax = plt.subplots()
	colors = sns.color_palette("BrBG", 9)
	ax.plot(Ns_Rlinks, Prob_a_Rlinks_bare, '-o', markersize=6, color=colors[2], label='Right of site, bareWLC')
	ax.plot(Ns_Rlinks, Prob_a_Rlinks_kinked, '--o', markersize=6, color=colors[1], label='Right of site, kinkedWLC')
	ax.plot(Ns_Llinks, Prob_a_Llinks_bare, '-o', markersize=6, color=colors[-3], label='Left of site, bareWLC')
	ax.plot(Ns_Llinks, Prob_a_Llinks_kinked, '--o', markersize=6, color=colors[-2], label='Left of site, kinkedWLC')
	plt.xlabel(r'Loop length, $N = R_{max} /2l_p$')
	plt.ylabel(f'Looping probability, a={a}nm')
	plt.subplots_adjust(bottom=0.15)
	plt.title(f'Kinked vs. Bare WLC with $l_p$={lp}nm, a={a}nm')

	if loglog:
		plt.xscale('log')
		plt.yscale('log')
		plt.legend(loc=(0.25, 0.025))
		plt.savefig(f'plots/loops/Sarah/lp{lp}nm_a{a}nm_kinked_vs_bare_loglog.png')
	else:
		plt.legend(loc=1)
		plt.subplots_adjust(right=0.95, left=0.16)
		plt.savefig(f'plots/loops/Sarah/lp{lp}nm_a{a}nm_kinked_vs_bare_linear.png')

def compute_looping_prob_matrix(a, lp=50):
	"""Compute full matrix of looping probabilities for all nucleosomes in chain.
	Returns two 51x51 matrices, one for 50 nucs right of and including nucleation site
	and one for 50 nucs left of and including nucleation site.

	Assumes chain looks like this:
	  35 . . . 35, 23, 15, 26, 30, 245, nucleation site, 47, 21, 18, 15, 20, 17, 35 . . . 35
	Nucleosome numbers:
	50 . . . 6,   5,  4,  3,  2,  1,          0        ,    1,  2,  3,  4,  5,  6,  7, . . . 50
	
	Each matrix element :math:`p_{ij}` is the probability that the ith and jth
	nucleosome will come into contact within a radius :math:`a` (in nm), where i and j 
	run from 0 (nucleation site) to 50. The diagonal elements :math:`p_{ii}=1`. The lower
	diagonal elements :math:`p_{ji}` were calculated explicitly for j=1 to 6. 

	For j>=7, extracted looping probabilities from homogenous 35bp linker chain, and
	assumed :math:`p_{ji}=p_{ji}` to fill in lower diagonals.
	
	By default, lp is 50nm.
	"""

	#Diagonal elements are probability 1 because chain length is 0.
	#shape is 51 by 51 because 50 nucs to right/left + nucleation site (51 nucs, 50 linkers)
	Rprobs = np.ones((len(Rlinks)+1, len(Rlinks)+1))
	Lprobs = np.ones((len(Llinks)+1, len(Llinks)+1))
	a_in_bp = a / ncg.dna_params['lpb']
	lp_in_bp = lp / ncg.dna_params['lpb']


	#only first 6nucs on both sides have heterogenous linker lengths
	max_hetero = 6
	#51 nucleosomes total, including nucleation site
	num_nucs = Rprobs.shape[0]
	for i in range(0, max_hetero):
		rlinks = Rlinks[i:]
		llinks = Llinks[i:]
		Rfilepath = Path(f'csvs/Bprops/0unwraps/heterogenous/Sarah/Rlinks_{i+1}to50nucs')
		Lfilepath = Path(f'csvs/Bprops/0unwraps/heterogenous/Sarah/Llinks_{i+1}to50nucs')
		#UPPER DIAGONAL
		if i==0:
			integral_R = np.load(Rfilepath/'kinkedWLC_greens_Rlinks_50nucs_1000rvals.npy')
			integral_L = np.load(Lfilepath/'kinkedWLC_greens_Llinks_50nucs_1000rvals.npy')
		else:
			integral_R = np.load(Rfilepath/f'kinkedWLC_greens_Rlinks_{i+1}to50nucs_{len(rlinks)}nucs.npy')
			integral_L = np.load(Lfilepath/f'kinkedWLC_greens_Llinks_{i+1}to50nucs_{len(llinks)}nucs.npy')

		rvals = np.linspace(0.0, 1.0, 1000)
		Prob_a_Rlinks = wlc.prob_R_in_radius_a_given_L(a_in_bp, integral_R, rvals, rlinks, unwrap)
		Rprobs[i, (i+1):] = Prob_a_Rlinks
		Prob_a_Llinks = wlc.prob_R_in_radius_a_given_L(a_in_bp, integral_L, rvals, llinks, unwrap)	
		Lprobs[i, (i+1):] = Prob_a_Llinks

		#LOWER DIAGONAL -- confirmed that it's nearly identical to upper diagonal, as in pij = pji
		if i < max_hetero - 1:
			rlinks = Rlinks_rev[i:]
			llinks = Llinks_rev[i:]
			Rfilepath = Path(f'csvs/Bprops/0unwraps/heterogenous/Sarah/Rlinks_{max_hetero-i}to1nucs')
			Lfilepath = Path(f'csvs/Bprops/0unwraps/heterogenous/Sarah/Llinks_{max_hetero-i}to1nucs')
			integral_R = np.load(Rfilepath/f'kinkedWLC_greens_Rlinks_{max_hetero-i}to1nucs_{max_hetero-i}nucs.npy')
			integral_L = np.load(Lfilepath/f'kinkedWLC_greens_Llinks_{max_hetero-i}to1nucs_{max_hetero-i}nucs.npy')
			Prob_a_Rlinks = wlc.prob_R_in_radius_a_given_L(a_in_bp, integral_R, rvals, rlinks, unwrap)
			Rprobs[max_hetero-i, 0:(max_hetero-i)] = Prob_a_Rlinks[::-1]
			Prob_a_Llinks = wlc.prob_R_in_radius_a_given_L(a_in_bp, integral_L, rvals, llinks, unwrap)	
			Lprobs[max_hetero-i, 0:(max_hetero-i)] = Prob_a_Llinks[::-1]

		#Rlinks1to1 nucs wasn't calculated because p01 = p10 by definition (only 1 linker)
		if i == max_hetero - 1:
			#equivalent to hardcoding Rprobs[1, 0] = Rprobs[0, 1]
			Rprobs[max_hetero-i, 0] = Rprobs[max_hetero-i-1, max_hetero-i]
			Lprobs[max_hetero-i, 0] = Lprobs[max_hetero-i-1, max_hetero-i]

	#6th nuc onward, extract looping probabilities from homogenous 35 bp chain
	greens35 = np.load('csvs/Bprops/0unwraps/35link/kinkedWLC_greens_35link_0unwraps_1000rvals_50nucs.npy')
	links35 = np.tile(35, 50)
	Prob_a_35bplinks = wlc.prob_R_in_radius_a_given_L(a_in_bp, greens35, rvals, links35, unwrap)
	for i in range(max_hetero, num_nucs):
		Rprobs[i, (i+1):] = Prob_a_35bplinks[0:(num_nucs-i-1)]
		if i > max_hetero:
			#7th nuc onwards, assume pij = pji to fill in rest of lower diagonal
			for j in range(0, i):
				Rprobs[i, j] = Rprobs[j, i]
				Lprobs[i, j] = Lprobs[j, i]

	#Save matrices in csv format
	Rdf = pd.DataFrame(Rprobs)
	Ldf = pd.DataFrame(Lprobs)
	Rdf.to_csv(f'csvs/Bprops/0unwraps/heterogenous/Sarah/looping_probs_a_{a}nm_50nucs_right_of_nucleation_site.csv')
	Ldf.to_csv(f'csvs/Bprops/0unwraps/heterogenous/Sarah/looping_probs_a_{a}nm_50nucs_left_of_nucleation_site.csv')
	return Rprobs, Lprobs

