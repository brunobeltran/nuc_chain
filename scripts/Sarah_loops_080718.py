r"""
Deepti Kannan
08-07-18

Analysis of looping probabilities for Sarah's heterogenous chain:
. . . 35, 23, 15, 26, 30, 245, nucleation site, 47, 21, 18, 15, 20, 17, 35 . . .
"""

import numpy as np
from matplotlib import pyplot as plt
from MultiPoint import propagator
from nuc_chain import fluctuations as wlc
from nuc_chain import geometry as ncg
from pathlib import Path
from scipy import stats

Klin = np.linspace(0, 10**5, 20000)
Klog = np.logspace(-3, 5, 10000)
Kvals = np.unique(np.concatenate((Klin, Klog)))
#convert to little k -- units of inverse bp (this results in kmax = 332)
kvals = Kvals / (2*wlc.default_lp)
links35 = np.tile(35, 44)
Rlinks = np.array([47, 21, 18, 15, 20, 17])
Llinks = np.array([245, 30, 26, 15, 23, 35])

#links to right of methylation site (50 in total)
Rlinks_50nucs = np.concatenate((Rlinks, links35))
#links to left of methylation site (50 in total)
Llinks_50nucs = np.concatenate((Llinks, links35))
unwrap = 0

#BARE WLC LOOPING PROBABILITY CALCULATIONS
Kprops_bareWLC = wlc.tabulate_bareWLC_propagators(Kvals)
rvals = np.linspace(0.0, 1.0, 1000)
qprop_R = wlc.bareWLC_gprop(kvals, Rlinks, unwrap, props=Kprops_bareWLC)
qintegral_R = wlc.BRN_fourier_integrand_splines(kvals, Rlinks, unwrap, Bprop=qprop_R, rvals=rvals) #default: 1000 rvals
qprop_L = wlc.bareWLC_gprop(kvals, Llinks, unwrap, props=Kprops_bareWLC)
qintegral_L = wlc.BRN_fourier_integrand_splines(kvals, Llinks, unwrap, Bprop=qprop_L, rvals=rvals) #default: 1000 rvals

#check probabilities are normalized
PRN_Rlinks = wlc.prob_R_given_L(qintegral_R, rvals, Rlinks, unwrap)
PRN_Llinks = wlc.prob_R_given_L(qintegral_L, rvals, Llinks, unwrap)

#Calculate looping with a 10th of a Kuhn length
a = 0.1 * 2 * wlc.default_lp #this corresponds to 10.0 nm
Prob_a_Rlinks = wlc.prob_R_in_radius_a_given_L(a, qintegral_R, rvals, Rlinks, unwrap)
Prob_a_Llinks = wlc.prob_R_in_radius_a_given_L(a, qintegral_L, rvals, Llinks, unwrap)

#want x axis to be N = Rmax / (2lp)
Rmax_Rlinks = wlc.Rmax_from_links_unwraps(Rlinks, unwraps=unwrap) #max WLC chain length in bp
Ns_Rlinks = Rmax_Rlinks / (2*wlc.default_lp)
Rmax_Llinks = wlc.Rmax_from_links_unwraps(Llinks, unwraps=unwrap) #max WLC chain length in bp
Ns_Llinks = Rmax_Llinks / (2*wlc.default_lp)

#Plot bare WLC looping probabilities to compare to Sarah's plot
fig, ax = plt.subplots()
ax.loglog(Ns_Rlinks, Prob_a_Rlinks, '-o', label='Rlinks')
ax.loglog(Ns_Llinks, Prob_a_Llinks, '-o', label='Llinks')
ax.legend()
plt.xlabel('Loop length, N')
plt.ylabel('Looping probability')

#Compare to looping probabilities from my kinked model

#for each chain, do fourier inversion integral and save output as .npy file in Bprops directory
Bprop_R = pickle.load(open(f'csvs/Bprops/{unwrap}unwraps/Sarah/B0_k_given_N_Rlinks_7nucs_30000Ks.p', 'rb'))
Bprop_L = pickle.load(open(f'csvs/Bprops/{unwrap}unwraps/Sarah/B0_k_given_N_Llinks_6nucs_30000Ks.p', 'rb'))
Rlinks = np.array([47, 21, 18, 15, 20, 17, 35])
Llinks = np.array([245, 30, 26, 15, 23, 35])
wlc.plot_BKN(Kvals, Bprop_R, Rlinks, 0, Nvals=np.arange(1, 7))
wlc.plot_BKN(Kvals, Bprop_L, Llinks, 0, Nvals=np.arange(1, 7))
rvals = np.linspace(0.0, 1.0, 1000)
integral_R = wlc.BRN_fourier_integrand_splines(kvals, Rlinks, unwrap, Bprop=Bprop_R, rvals=rvals) #default: 1000 rvals
integral_L = wlc.BRN_fourier_integrand_splines(kvals, Llinks, unwrap, Bprop=Bprop_L, rvals=rvals) #default: 1000 rvals

#integral takes ~10 min to run, so prob worth saving
np.save(f'csvs/Bprops/{unwrap}unwraps/Sarah/kinkedWLC_greens_Rlinks_7nucs_{len(rvals)}rvals.npy', integral_R, allow_pickle=False)
np.save(f'csvs/Bprops/{unwrap}unwraps/Sarah/kinkedWLC_greens_Llinks_6nucs_{len(rvals)}rvals.npy', integral_L, allow_pickle=False)
wlc.plot_greens(integral_R, Rlinks, unwrap, Nvals=np.arange(1, 8))
wlc.plot_greens(integral_L, Llinks, unwrap, Nvals=np.arange(1, 8))

#Calculate looping probabilities for Sarah:
Prob_a_Rlinks_kinked = wlc.prob_R_in_radius_a_given_L(a, integral_R, rvals, Rlinks, unwrap)
Prob_a_Llinks_kinked = wlc.prob_R_in_radius_a_given_L(a, integral_L, rvals, Llinks, unwrap)

#Compare to her current values, which assume bare WLC with lp = 10.18nm
lp = 10.18 / ncg.dna_params['lpb']
qprop_R_lp10 = wlc.bareWLC_gprop(kvals, Rlinks_50nucs, unwrap, props=Kprops_bareWLC, lp=lp)
qintegral_R_lp10 = wlc.BRN_fourier_integrand_splines(kvals, Rlinks, unwrap, Bprop=qprop_R_lp10, rvals=rvals) #default: 1000 rvals
qprop_L_lp10 = wlc.bareWLC_gprop(kvals, Llinks_50nucs, unwrap, props=Kprops_bareWLC, lp=lp)
qintegral_L_lp10 = wlc.BRN_fourier_integrand_splines(kvals, Llinks, unwrap, Bprop=qprop_L_lp10, rvals=rvals) #default: 1000 rvals

#rescale N; plot these 4 things correctly
a = 0.1 * 2 * wlc.default_lp
Prob_a_Rlinks_lp10 = wlc.prob_R_in_radius_a_given_L(a, qintegral_R_lp10, rvals, Rlinks_50nucs, unwrap)
Prob_a_Llinks_lp10 = wlc.prob_R_in_radius_a_given_L(a, qintegral_L_lp10, rvals, Llinks_50nucs, unwrap)
Ns_Rlinks_lp10 = Rmax_Rlinks / (2*lp)
Ns_Llinks_lp10 = Rmax_Llinks / (2*lp)
Ns_Rlinks_lp10_7nucs = Ns_Rlinks_lp10[0:7]
Ns_Llinks_lp10_6nucs = Ns_Llinks_lp10[0:6]

#Bare WLC lp = 50 nm vs. Kinked WLC
fig, ax = plt.subplots()
ax.loglog(Ns_Rlinks, Prob_a_Rlinks, '-o', color = np.random.rand(3), label='Right of nucleation site, bareWLC')
ax.loglog(Ns_Rlinks, Prob_a_Rlinks_kinked_50nucs, '--o', color = np.random.rand(3), label='Right of nucleation site, kinkedWLC')
ax.loglog(Ns_Llinks, Prob_a_Llinks, '-o', color = np.random.rand(3), label='Left of nucleation site, bareWLC')
ax.loglog(Ns_Llinks, Prob_a_Llinks_kinked_50nucs, '--o', color = np.random.rand(3), label='Left of nucleation site, kinkedWLC')
ax.legend()
plt.xlabel('Loop length, N')
plt.ylabel('Looping probability, a = 0.1')
plt.title('Kinked vs. Bare WLC with $l_p$=50nm')

#Bare WLC lp = 10.18 nm vs. Kinked WLC
fig, ax = plt.subplots()
ax.loglog(Ns_Rlinks_lp10, Prob_a_Rlinks_lp10, '-o', color = np.random.rand(3), label='Right of nucleation site, bareWLC')
ax.loglog(Ns_Rlinks_lp10, Prob_a_Rlinks_kinked_50nucs, '--o', color = np.random.rand(3), label='Right of nucleation site, kinkedWLC')
ax.loglog(Ns_Llinks_lp10, Prob_a_Llinks_lp10, '-o', color = np.random.rand(3), label='Left of nucleation site, bareWLC')
ax.loglog(Ns_Llinks_lp10, Prob_a_Llinks_kinked_50nucs, '--o', color = np.random.rand(3), label='Left of nucleation site, kinkedWLC')
ax.legend()
plt.xlabel('Loop length, N')
plt.ylabel('Looping probability, a = 0.1')
plt.title('Kinked vs. Bare WLC with $l_p$=10.18nm')


#Calculate slope of long chain Gaussian limit; should scale as N^(-3/2)
m = stats.linregress(np.log(Ns_Llinks_lp10[8:]), np.log(Prob_a_Llinks_lp10[8:]))[0]


### Convert above analysis into decomposed functions that can be reused upon later analyses ###

def load_variables_for_Sarah_looping():
	"""Run this function when starting a new ipython environment. Keep the variables in the
	environment since all future functions will use them."""
	Klin = np.linspace(0, 10**5, 20000)
	Klog = np.logspace(-3, 5, 10000)
	Kvals = np.unique(np.concatenate((Klin, Klog)))
	#convert to little k -- units of inverse bp (this results in kmax = 332)
	kvals = Kvals / (2*wlc.default_lp)
	links35 = np.tile(35, 44)
	Rlinks = np.array([47, 21, 18, 15, 20, 17])
	Llinks = np.array([245, 30, 26, 15, 23, 35])

	#links to right of methylation site (50 in total)
	Rlinks = np.concatenate((Rlinks, links35))
	#links to left of methylation site (50 in total)
	Llinks = np.concatenate((Llinks, links35))
	unwrap = 0
	#Bare WLC green's functions (1000 rvals, 50 nucs) for lp = 50nm
	qintegral_R_lp50 = np.load('csvs/Bprops/0unwraps/Sarah/bareWLC_greens_Rlinks_50nucs.npy')
	qintegral_L_lp50 = np.load('csvs/Bprops/0unwraps/Sarah/bareWLC_greens_Llinks_50nucs.npy')
	#Bare WLC green's functions (1000 rvals, 50 nucs) for lp = 10.18nm
	qintegral_R_lp10 = np.load('csvs/Bprops/0unwraps/Sarah/bareWLC_greens_Rlinks_50nucs_lp10.npy')
	qintegral_L_lp10 = np.load('csvs/Bprops/0unwraps/Sarah/bareWLC_greens_Llinks_50nucs_lp10.npy')
	#Kinked WLC green's function (1000 rvals, 50 nucs)
	integral_R = np.load('csvs/Bprops/0unwraps/Sarah/kinkedWLC_greens_Rlinks_50nucs_1000rvals.npy')
	integral_L = np.load('csvs/Bprops/0unwraps/Sarah/kinkedWLC_greens_Llinks_50nucs_1000rvals.npy')

#Run load_variables_for_Sarah_looping() before running any of these functions
def bareWLC_greens(lpvals, props=None):
	if props is None:
		props = wlc.tabulate_bareWLC_propagators(Kvals)

	for lp in lpvals:
		lp = lp / ncg.dna_params['lpb']
		Rfile = Path(f'csvs/Bprops/0unwraps/Sarah/bareWLC_greens_Rlinks_50nucs_lp{lp:.0f}.npy')
		if (Rfile.is_file() is False):
			#bare WLC propagators
			qprop_R = wlc.bareWLC_gprop(kvals, Rlinks, unwrap, props=props, lp=lp)
			#bare WLC greens
			qintegral_R = wlc.BRN_fourier_integrand_splines(kvals, Rlinks, unwrap, Bprop=qprop_R)
			np.save(Rfile, qintegral_R)
		Lfile = Path(f'csvs/Bprops/0unwraps/Sarah/bareWLC_greens_Llinks_50nucs_lp{lp:.0f}.npy')
		if (Lfile.is_file() is False):
			#bare WLC propagators
			qprop_L = wlc.bareWLC_gprop(kvals, Llinks, unwrap, props=props, lp=lp)
			#bare WLC greens
			qintegral_L = wlc.BRN_fourier_integrand_splines(kvals, Llinks, unwrap, Bprop=qprop_L)
			np.save(Lfile, qintegral_L)
