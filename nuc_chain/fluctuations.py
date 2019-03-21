# -*- coding: utf-8 -*-
r"""Kinked WLC with twist and fixed ends

This module calculates statistics for a series of worm-like chains with twist (DNA linkers) connected
by kinks imposed by nucleosomes. Calculations include R^2, Kuhn length, propogator matrices,
full Green's function for the end-to-end distance of the polymer, and looping statistics. For a detailed
derivation of how these calculations are carried out, refer to Deepti's notes.

#TODO move anything matching "def plot" to the plotting module (or testing
module), adjust code to reference correct members of fluctuations module.
"""
import re
import inspect

import pickle
import numpy as np
import scipy
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy import sparse
from scipy import special
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.interpolate import splint
from scipy.interpolate import splrep
from . import utils
from . import wignerD as wd
from . import geometry as ncg
from . import rotations as ncr
from . import data as ncd
from . import linkers as ncl
from .linkers import convert
from MultiPoint import propagator
from MultiPoint import WLCgreen
from memory_profiler import profile
from multiprocessing import Pool
from filelock import Timeout, FileLock
import sys

###{{{
# """Constants"""

tau_d_nm = ncg.dna_params['tau_d']/ncg.dna_params['lpb']
"""naked DNA twist density in rad/nm"""
default_lt = 100 / ncg.dna_params['lpb']
"""twist persistance length of DNA in bp: 100 nm, or ~301 bp"""
default_lp = 50 / ncg.dna_params['lpb']
"""bend persistance length of DNA in bp: 50 nm, or ~150 bp"""

Klin = np.linspace(0, 10**5, 20000)
Klog = np.logspace(-3, 5, 10000)
Kvals = np.unique(np.concatenate((Klin, Klog)))
#convert to little k -- units of inverse bp (this results in kmax = 332)
kvals = Kvals / (2*default_lp)
"""Good values to use for integrating our Green's functions. If the lp of the
bare chain under consideration changes, these should change."""
###}}}

###{{{
# R^2 and Kuhn length calculations

def build_B_matrices_for_R2(link, alpha, beta, gamma, lt=default_lt, lp=default_lp, tau_d=ncg.dna_params['tau_d'], lmax=2):
    r"""Helper function to construct propogator B matrices for a single linker length link with kink
    rotation given by alpha, beta, gamma. Returns the following three matrices:

    .. math::

        B^{(n)} = \lim_{k \to 0}\frac{d^n B}{dk^n}

    for n = 0, 1, and 2. where B is defined as

    .. math::

        B^{l_f j_f}_{l_0 j_0} = \sqrt{\frac{8\pi^2}{2l_f+1}} {\mathcal{D}}^{j_f j_0}_{l_f}(-\gamma, -\beta, -\alpha) g^{j_0}_{l_f l_0}
        B^{(n)}[I_f, I_0] = M[I_f, I_0] * g^{(n)}[I_f, I_0]

    The g matrix is used for the linker propogator, and the M matrix represents the rotation due to the kink.
    All matrices are super-indexed by B[If, I0] where :math:`I(l, j) = l^2 + l + j`. :math:`I` can take on :math:`l^2+2l+1` possible values.

    Notes
    -----
    Andy adds 1 to the above formula for :math:`I` since his script is in Matlab, which is
    one-indexed.

    Parameters
    ----------
    link : float
        linker length in bp
    lt : float
        twist persistence length in bp
    lp : float
        DNA persistence length in bp
    tau_d : float
        twist density of naked DNA in rad/bp
    lmax : int
        maximum eigenvalue l for which to compute wigner D' (default lmax = 2)

    Returns
    -------
    [B0, B1, B2] : (3,) list
        list of 3 matrices, each of dimension :math:`l_{max}^2 + 2l_{max} + 1`
    """

    # corresponds to If, I0 indices in Andy's notes: for every l, (2l+1) possible values of j
    ntot = lmax**2 + 2*lmax + 1
    # so matrix elements look like g[If,I0] where I0 = l0**2 + l0 + j0 and If = lf**2 + lf + jf
    # Note that for linker propogators (g matrices), jf always equals j0 (only perturbs l)
    # and for kink propogators (M matrices), lf always equals l0 (only perturbs j)
    # NOTE: for python, need to subtract 1 from Andy's formulas for indexing to work

    # g0 represents the 0th derivative of g with respect to k in the limit as k goes to 0
    g0 = np.zeros((ntot, ntot), 'complex')
    g1 = np.zeros_like(g0)
    g2 = np.zeros_like(g0)
    M  = np.zeros_like(g0)
    mywd = wd.wigner_d_vals()

    # define useful lambda functions of l and j used to compute eigenvalues, matrix elements
    I   = lambda l, j: l**2 + l + j # indexing
    al  = lambda l, j: np.sqrt((l-j)*(l+j)/(4*l**2 - 1)) # ladder coefficients alpha
    lam = lambda l, j: (l*(l+1))/(2*lp) + 0.5*((1/lt)-(1/lp))*j**2 - 1j*tau_d*j # eigenvalue of H0

    # build g and M matrices by looping over l0 and j0
    for l0 in range(lmax+1):
        for j0 in range(-l0, l0+1):
            # for this particular tuple (l0, j0), compute the relevant index in the g matrix:
            I0 = I(l0, j0)

            # Compute relevant values of lambda_lj and alpha_lj to construct g0, g1, g2
            laml0   = lam(l0, j0)
            laml0p1 = lam(l0+1, j0)
            laml0m1 = lam(l0-1, j0)
            laml0p2 = lam(l0+2, j0)
            laml0m2 = lam(l0-2, j0)
            all0    = al(l0, j0)
            all0p1  = al(l0+1, j0)
            # NOTE: this will produce nans for (l0, j0) = (2, 2) and (2, -2), but
            # this quantity isn't used for those values of l0, j0 so no problem
            if (l0 != 2 or abs(j0) != 2):
                all0m1  = al(l0-1, j0)
            all0p2  = al(l0+2, j0)

            ### Construct g0 matrix###

            # answer for g0 says l = l0 due to delta function, so g0 is diagonal
            g0[I0, I0] = np.exp(-laml0*link)

            #### Construct g1 matrix###

            # first consider case where l = l0-1
            l = l0 - 1
            If = I(l, j0)
            # check out of bounds for l, ensure l does not exceed j0 (because j = j0)
            if (l >= 0) and (l <= lmax) and (l >= np.abs(j0)):
                g1[If, I0] = (1j*all0/(laml0m1 - laml0))*(np.exp(-laml0*link) - np.exp(-laml0m1*link))

            # next consider the case where l = l0+1
            l = l0 + 1
            If = I(l, j0)
            if (l >= 0) and (l <= lmax) and (l >= np.abs(j0)):
                g1[If, I0] = (1j*all0p1/(laml0p1 - laml0))*(np.exp(-laml0*link) - np.exp(-laml0p1*link))

            #### Construct g2 matrix###

            # Case 1: l = l0 + 2
            l = l0 + 2
            If = I(l, j0)
            # only valid case is when (l0, j0) = (0, 0)
            if (l >= 0) and (l <= lmax) and (l >= np.abs(j0)):
                g2[If, I0] = -2*all0p1*all0p2*(np.exp(-laml0*link)/((laml0p1 - laml0)*(laml0p2 - laml0)) +
                                              np.exp(-laml0p1*link)/((laml0 - laml0p1)*(laml0p2 - laml0p1)) +
                                              np.exp(-laml0p2*link)/((laml0 - laml0p2)*(laml0p1 - laml0p2)))

            # Case 2: l = l0 --- diagonal entries of g2 matrix
            l = l0
            If = I(l, j0)
            # Case 2A: terms with l0 + 1
            if (l >= 0) and (l <= lmax) and (l >= np.abs(j0)):
                g2[If, I0] = -2*all0p1**2*(link*np.exp(-laml0*link)/(laml0p1 - laml0) +
                                          (np.exp(-laml0p1*link) - np.exp(-laml0*link))/(laml0 - laml0p1)**2)
            # Case 2B: terms with l0 - 1
            if (l >= 0) and (l <= lmax) and (l >= np.abs(j0)+1):
                g2[If, I0] += -2*all0**2*(link*np.exp(-laml0*link)/(laml0m1 - laml0) +
                                          (np.exp(-laml0m1*link) - np.exp(-laml0*link))/(laml0 - laml0m1)**2)

            # Case 3: l = l0 - 2
            l = l0 - 2
            If = I(l, j0)
            # only valid case is when (l0, j0) = (2, 0)
            if (l >= 0) and (l <= lmax) and (l >= np.abs(j0)):
                g2[If, I0] = -2*all0*all0m1*(np.exp(-laml0*link)/((laml0m1 - laml0)*(laml0m2 - laml0)) +
                                              np.exp(-laml0m1*link)/((laml0 - laml0m1)*(laml0m2 - laml0m1)) +
                                              np.exp(-laml0m2*link)/((laml0 - laml0m2)*(laml0m1 - laml0m2)))

            # Next build M matrix
            for jf in range(-l0, l0+1):
                If = I(l0, jf)
                M[If, I0] = mywd.get(l0, jf, j0, -gamma, -beta, -alpha) / mywd.normalize(l0, jf, j0)

    B0 = M@g0
    B1 = M@g1
    B2 = M@g2
    return [B0, B1, B2]


def r2wlc(ldna, lp=default_lp):
    """Analytical formula for R^2 of WLC as a function of length of chain."""
    return 2*(lp*ldna - lp**2 + lp**2*np.exp(-ldna/lp))

def R2_kinked_WLC_no_translation(links, figname='fig', plotfig=False,
        lt=default_lt, lp=default_lp, w_ins=ncg.default_w_in,
        w_outs=ncg.default_w_out, tau_d=ncg.dna_params['tau_d'],
        tau_n=ncg.dna_params['tau_n'], lmax=2, helix_params=ncg.helix_params_best,
        unwraps=None, random_phi=False):
    """Calculate the mean squared end-to-end distance, or :math:`\langle{R^2}\rangle` of a kinked WLC with a given set of linkers and unwrapping amounts.

    Parameters
    ----------
    links : (L,) array-like
        linker length in bp
    figname: string
        name of figure to be saved as pdf
    plotfig: bool (default = False)
        whether or not to plot R^2 vs. Rmax
    w_ins : float or (L+1,) array_like
        amount of DNA wrapped on entry side of central dyad base in bp
    w_outs : float or (L+1,) array_like
        amount of DNA wrapped on exit side of central dyad base in bp
    tau_n : float
        twist density of nucleosome-bound DNA in rad/bp
    tau_d : float
        twist density of naked DNA in rad/bp
    lt : float
        twist persistence length in bp
    lp : float
        DNA persistence length in bp
    lmax : int
        maximum eigenvalue l for which to compute wigner D' (default lmax = 2)

    Returns
    -------
    r2 : (L,) array-like
        mean square end-to-end distance of kinked chain as a function of chain length in nm^2
    ldna : (L,) array-like
        mean square end-to-end distance of kinked chain as a function of chain length in nm
    kuhn : float
        Kuhn length as defined by :math:`\langle{R^2}\rangle / R_{max}` in long chain limit
    """
    b = helix_params['b']
    num_linkers = len(links)
    num_nucleosomes = num_linkers + 1
    w_ins, w_outs = convert.resolve_wrapping_params(unwraps, w_ins, w_outs, num_nucleosomes)
    # calculate unwrapping amounts based on w_ins and w_outs
    mu_ins = (b - 1)/2 - w_ins
    mu_outs = (b - 1)/2 - w_outs
    # only need one g matrix per linker length, no need to recalculate each time
    # perhaps we tabulate all g's and all M's and then mix and match to grow chain?
    # for now, build dictionary of (link, wrapping) -> [B0, B1, B2]
    bmats = {}

    # B0-2curr will keep track of the B matrices as they propogate along the chain
    # initialize based on very first linker in chain
    link = mu_outs[0] + links[0] + mu_ins[1]
    wrap = w_outs[0] + w_ins[1]
    key = (link, wrap)
    R = ncg.OmegaE2E(wrap, tau_n=tau_n)
    # recall that our OmegaE2E matrix is designed to be applied from the right
    # so in order to add an arbitrary twist *before* the action of the
    # nucleosome (as if from changing the linker length) then we should apply
    # Rz to the left of R so that when the combined R is applied on the *right*
    # then the extra Rz is applied "first".
    # in this code, we use "(-gamma, -beta, -alpha)" from the left as a proxy
    # from right multiplication in build_B_matrices_for_R2
    if random_phi:
        R = ncr.Rz(2*np.pi*np.random.rand()) @ R
    alpha, beta, gamma = ncr.zyz_from_matrix(R)
    bmats[key] = build_B_matrices_for_R2(link, alpha, beta, gamma, lt, lp, tau_d, lmax)
    B0curr, B1curr, B2curr = bmats[key]

    # calculate R^2 as a function of number of nucleosomes (r2[0] is 0 nucleosomes)
    r2 = np.zeros((num_linkers,))
    lengthDNA = np.zeros_like(r2)
    r2[0] = 3*np.real(B2curr[0,0]/B0curr[0, 0])
    lengthDNA[0] = link

    # recursively calculate Nth propagator using B matrices
    for i in range(1, num_linkers):
        # add up the effective linker lengths including unwrapping
        link = mu_outs[i] + links[i] + mu_ins[i+1]
        # w_ins[i+1] because the ith linker is between i, and i+1 nucs
        wrap = w_outs[i] + w_ins[i+1]
        key = (link, wrap)
        # update dictionary for this linker and wrapping amount, if necessary
        if key not in bmats:
            R = ncg.OmegaE2E(wrap, tau_n=tau_n)
            if random_phi:
                R = ncr.Rz(2*np.pi*np.random.rand()) @ R
            alpha, beta, gamma = ncr.zyz_from_matrix(R)
            bmats[key] = build_B_matrices_for_R2(link, alpha, beta, gamma, lt, lp, tau_d, lmax)
        B0next, B1next, B2next = bmats[key]

        # propogate B0curr, B1curr, B2curr matrices by a linker
        B0temp = B0next@B0curr
        B1temp = B1next@B0curr + B0next@B1curr
        B2temp = B2next@B0curr + 2*B1next@B1curr + B0next@B2curr
        B0curr = B0temp
        B1curr = B1temp
        B2curr = B2temp

        # Rz^2 is B2[0,0], so multiply by 3 to get R^2
        # divide by B0[0,0] to ensure normalization is OK (shouldn't matter, since B0[0,0] is 1)
        r2[i] = 3*np.real(B2curr[0,0]/B0curr[0, 0])
        lengthDNA[i] = lengthDNA[i-1] + link

    # take absolute value of R2, convert to nm^2
    r2 = np.abs(r2) * (ncg.dna_params['lpb'])**2
    lengthDNA = lengthDNA * ncg.dna_params['lpb']

    # Find scaling of R2 with length of chain at larger length scales to calculate Kuhn length
    kuhn = stats.linregress(lengthDNA[5000:], r2[5000:])[0]
    # Plot r2 vs. length of chain (in bp)
    if plotfig:
        raise NotImplementedError('This module no longer plots.')
        # plt.figure()
        # plt.loglog(lengthDNA, np.abs(r2))
        # plt.xlabel('Length of chain in basepairs')
        # plt.ylabel(r'$<R^2>$')
        # plt.savefig(figname)

    return r2, lengthDNA, kuhn

def tabulate_kuhn_lengths():
    """Calculate and save Kuhn lengths for chains with fixed linker lengths 1-250 bp
    and unwrapping amounts 0-146 bp. Saves output in 'kuhns_1to250links_0to146unwraps.npy' file.

    Returns
    -------
    kuhns : (250, 147) array-like
        Kuhn length in nm for a nucleosome chain with fixed linkers and unwrapping amount
    """

    links, unwraps = np.mgrid[1:251, 0:147]
    kuhns = np.zeros_like(links).astype(float)
    # also save dictionary (link, wrap) -> [B0, B1, B2] since we have to compute all tuples for this code anyway
    Bmats = {}

    for i in range(links.shape[0]):
        linkers = np.tile(links[i, 0], 10000)
        for j in range(unwraps.shape[1]):
            r2, ldna, kuhn, bmats = R2_kinked_WLC_no_translation(linkers, unwraps=unwraps[i, j])
            kuhns[i, j] = kuhn
            # append dictionary from previous calculation to the global dictionary
            Bmats = {**Bmats, **bmats}
        pickle.dump(kuhns, open('kuhns_1to250links_0to146unwraps.p', 'wb'))
        print(f'Calculated, saved Kuhn length for link={links[i,0]}bp')

    pickle.dump(Bmats, open('B0B1B2matrices_key_wrap.p', 'wb'))
    # also save in .npy format just in case
    np.save('kuhns_1to250links_0to146unwraps', kuhns)
    return kuhns

def tabulate_kuhn_lengths_straight_linkers_genomic_distance():
    """Calculate Kuhn length in nm^2/bp for limiting case where all linkers
    are straight (no kinks) but the length of the chain still includes the 146
    bp bound to each nucleosome (plus the bare linkers). All this does is
    rescale the answer we get for fully unwrapped linkers (which also
    corresponds to no kinks) but reduces the chain length to fit this problem.
    """

    links, unwraps = np.mgrid[1:251, 0:147]
    kuhns = np.load('csvs/kuhns_1to250links_0to146unwraps.npy')
    b = ncg.helix_params_best['b']

    #extract kuhn length for straight linkers (fully unwrapped), this is just
    #100 nm for all 250 linkers
    kuhns_straight_linkers = kuhns[:, 146]
    kuhns_corrected = np.zeros_like(kuhns)

    for i, link in enumerate(links[:,0]):
        for j, unwrap in enumerate(unwraps[0, :]):
            Lunbound = link + unwrap
            Lbound = b - unwrap - 1
            scaling_factor = Lunbound / (Lunbound + Lbound)
            #rescale 100nm based on unwrapping and linker length
            kuhns_corrected[i, j] = kuhns_straight_linkers[i] * scaling_factor * ncg.dna_params['lpb']

    #now in units of nm^2 / bp
    return kuhns_corrected

def plot_kuhn_lengths_for_different_unwrapping():
    """Plots kuhn length in nm^2/bp as a function of linker length, with 2
    different unwrapping levels: 0, and 20 bp. Also plots upper limit of
    Kuhn length (straight linkers)."""

    kuhns_nm_squared_per_bp = tabulate_kuhn_lengths_in_genomic_distance()
    kuhns_straight_linkers = (
    tabulate_kuhn_lengths_straight_linkers_genomic_distance())

    #first plot just the kuhn lengths for 0, 20 bp unwrapping
    linkers = np.arange(1, 251)
    kuhns_no_unwrapping = plt.plot(linkers, kuhns_nm_squared_per_bp[:, 0],
                                   '-o', color='g')
    kuhns_no_unwrapping_straight = plt.plot(linkers,
                                            kuhns_straight_linkers[:,0], '--',
                                            color='g')
    kuhns_20bp_unwrapping = plt.plot(linkers, kuhns_nm_squared_per_bp[:, 20],
                                     '-o', markersize=4, color='r')

    kuhns_20bp_unwrapping_straight = plt.plot(linkers,
                                            kuhns_straight_linkers[:,20], '--',
                                            color='r')
    plt.legend(['No unwrapping', 'No unwrapping straight linkers',
                '20bp unwrapping', '20bp unwrapping straight linkers'])
    plt.xlabel('Linker length (bp)')
    plt.ylabel('Kuhn length (${nm}^2/bp$)')
    plt.title('Kuhn length for different unwrapping amounts, linkers 1-250 bp')

def tabulate_kuhn_lengths_in_nm2_per_bp():
    """Calculate Kuhn lengths for chains with fixed linker lengths 1-250 bp
    and unwrapping amounts 0-146 bp, where :math:`b = \langle{R^2}\rangle/R_{max}`
    and :math:`R_{max}` is total chain length in genomic distance (bp), including
    bp bound to nucleosome.

    Returns
    -------
    kuhns : (250, 147) array-like
        Kuhn length in nm^2/bp for each linker length and unwrapping amount
    """
    links, unwraps = np.mgrid[1:251, 0:147]
    kuhns = np.load('csvs/kuhns_1to250links_0to146unwraps.npy')
    b = ncg.helix_params_best['b']

    for i, link in enumerate(links[:, 0]):
        for j, unwrap in enumerate(unwraps[0, :]):
            Lunbound = link + unwrap
            Lbound = b - unwrap - 1
            scaling_factor = Lunbound / (Lunbound + Lbound)
            kuhns[i, j] *= scaling_factor * ncg.dna_params['lpb']
            ### ALTERNATIVE: get rid of tabulate_kuhn_lengths_in_bp() function and just do unit conversion:
            #genomic_bp_per_linker_nm =  (Lunbound + Lbound) / (Lunbound * ncg.dna_params['lpb'])
            # rescale kuhn length and convert Rmax back to bp
            #kuhns[i, j] *= genomic_bp_per_linker_nm

    # now in units of nm^2 / bp
    return kuhns

def tabulate_kuhn_lengths_in_genomic_distance():
    """Calculate Kuhn lengths for chains with fixed linker lengths 1-250 bp
    and unwrapping amounts 0-146bp, where :math:`b = (b in nm)^2 / (b in nm^2/bp)`.
    In other words, find the number of bp in a Kuhn length (includign wrapped
    base pairs)."""

    links, unwraps = np.mgrid[1:251, 0:147]
    kuhns = np.load('csvs/kuhns_1to250links_0to146unwraps.npy')
    b = ncg.helix_params_best['b']

    for i, link in enumerate(links[:, 0]):
        for j, unwrap in enumerate(unwraps[0, :]):
            Lunbound = link + unwrap
            Lbound = b - unwrap - 1
            genomic_bp_per_linker_nm =  (Lunbound + Lbound) / (Lunbound * ncg.dna_params['lpb'])
            # rescale kuhn length and convert Rmax back to bp
            kuhns[i, j] *= genomic_bp_per_linker_nm

    # now in units of bp
    return kuhns

def tabulate_kuhn_lengths_along_screw_axis():
    """Rescale kuhn length by (linker length nm / rise per linker in nm) so that
    length of polymer is measured along helical axis as opposed to along chain itself.

    Currently it rescales kuhn length by the purely geometrical rise/bp for
    that specific homogenous chain. Maybe in the future, we could find a better
    number for the "rise" that better accounts for the fluctuating chains real
    effective "screw" axis. """

    links, unwraps = np.mgrid[10:200, 0:147]
    kuhns = np.load('csvs/kuhns_1to250links_0to146unwraps.npy')
    link_ix, unwrap_ix, rise, angle, radius = ncg.tabulate_rise()
    b = ncg.helix_params_best['b']

    for i, link in enumerate(links[:, 0]):
        for j, unwrap in enumerate(unwraps[0, :]):
            Lunbound = link + unwrap
            scaling_factor = (Lunbound * ncg.dna_params['lpb']) / rise[i, j]
            kuhns[i+9, j] *= scaling_factor

    return kuhns

def heterogenous_chains_kuhn_lengths(links, unwraps=0, numiter=10, **kwargs):
    """Calculate Kuhn length for a heterogenous chain that samples uniformly
    from the linker lengths in 'links'. Performs calculation 'numiter' times,
    returns kuhn lengths from each calculation. Also compares results
    to harmonic average of corresponding homogenous chains. Use to test
    harmonic averaging rule."""

    #save a kuhn length for each iteration
    kuhns = np.zeros(numiter)
    r2 = np.zeros((numiter, 7500)) #grow chain to 7500 monomers, should be plenty
    ldna = np.zeros_like(r2)
    for i in range(numiter):
        links = np.random.choice(links, 7500)
        r2d, ldnad, kuhnsd = R2_kinked_WLC_no_translation(links, plotfig=False, unwraps=unwraps)
        r2[i, :] = r2d
        ldna[i, :] = ldnad
        kuhns[i] = kuhnsd

    print(f'Mean Kuhn length of {numiter} random chains: {np.mean(kuhns):.2f}nm')

    #calculate harmonic mean of kuhn lengths of corresponding fixed linker chains
    kuhns_1to250links_0to146unwraps = np.load('csvs/kuhns_1to250links_0to146unwraps.npy')
    harmonic_avg = np.mean(1/kuhns_1to250links_0to146unwraps[links-1, unwraps])**(-1)
    print(f'Harmonic mean kuhn length of fixed linker chains: {harmonic_avg:.2f}nm')
    arithmetic_avg = np.mean(kuhns_1to250links_0to146unwraps[links-1, unwraps])
    print(f'Mean kuhn length of fixed linker chains: {arithmetic_avg:.2f}nm')
    return kuhns

def harmonic_avg_exponential_kuhn_lengths(kuhns, links, mu):
    """Calculate harmonic averaged kuhn length with exponential weights
    determined by mean mu. Derivation of formula in Deepti's notes."""
    sum = (1/kuhns)@np.exp(-mu*links)
    prefactor = np.exp(mu)*(1 - np.exp(-mu))/(1 - np.exp(-mu*max(links)))
    kuhn_avg_inverse = sum*prefactor
    return (1/kuhn_avg_inverse)

def tabulate_r2_heterogenous_fluctuating_chains_by_variance(num_chains, chain_length, sigmas, mu=35, pool_size=None, **kwargs):
    """Tabulate R^2 for fluctuating heterogenous chains with increasing
    variance. "Box" variance model. Pass unwrapping parameters through kwargs."""
    n_sig = len(sigmas)
    links = np.zeros((n_sig, num_chains, chain_length-1))
    #For now, assume the same unwrapping amounts for all chains
    #w_ins, w_outs = convert.resolve_wrapping_params(unwraps, w_ins, w_outs, chain_length)
    for i in range(n_sig):
        links[i,:,:] = ncl.fake_linkers_increasing_variance(mu, sigmas[i], size=(num_chains,chain_length-1), type='box')
    rmax = np.zeros((n_sig, num_chains, chain_length))
    r2 = rmax.copy()
    variance = rmax.copy()
    chain_id = rmax.copy()
    kuhns = rmax.copy()
    def given_ij(ij):
        i, j = ij
        #note r2, ldna returned by R2_kinked_WLC() is the same shape as links. add a 0 to the beginning
        #to match bruno's code; Output is in nm
        R2, Rmax, kuhn = R2_kinked_WLC_no_translation(links[i,j,:].flatten(), plotfig=False, **kwargs)
        r2[i,j] = np.concatenate(([0], R2))
        rmax[i,j] = np.concatenate(([0], Rmax))
        variance[i] = sigmas[i]
        kuhns[i,j] = kuhn
        chain_id[i,j] = j
    if pool_size is None:
        for i in range(n_sig):
            for j in range(num_chains):
                given_ij((i,j))
    else:
        with Pool(processes=pool_size) as p:
            p.map(given_ij, [(i,j) for i in range(n_sig) for j in range(num_chains)])
    df = pd.DataFrame(np.stack([
        r2.flatten(), rmax.flatten(), variance.flatten(), chain_id.flatten(), kuhns.flatten()
        ], axis=1), columns=['r2', 'rmax', 'variance', 'chain_id', 'kuhn'])
    return df

def tabulate_r2_heterogenous_fluctuating_chains_exponential(num_chains, chain_length, mu=35, pool_size=None, **kwargs):
    """Tabulate R^2 for fluctuating heterogenous chains with increasing variance. Pass unwrapping parameters
    through kwargs."""
    links = np.zeros((num_chains, chain_length-1))
    #For now, assume the same unwrapping amounts for all chains
    #w_ins, w_outs = convert.resolve_wrapping_params(unwraps, w_ins, w_outs, chain_length)
    links = ncl.independent_linker_lengths(mu, size=(num_chains,chain_length-1))
    rmax = np.zeros((num_chains, chain_length))
    r2 = rmax.copy()
    chain_id = rmax.copy()
    kuhns = rmax.copy()
    def given_chain_i(i):
        #note r2, ldna returned by R2_kinked_WLC() is the same shape as links. add a 0 to the beginning
        #to match bruno's code; Output is in nm
        R2, Rmax, kuhn = R2_kinked_WLC_no_translation(links[i,:].flatten(), plotfig=False, **kwargs)
        r2[i] = np.concatenate(([0], R2))
        rmax[i] = np.concatenate(([0], Rmax))
        kuhns[i] = kuhn
        chain_id[i] = i
    if pool_size is None:
        for i in range(num_chains):
            given_chain_i(i)
    else:
        raise NotImplementedError('No Pool here plz.')
        # with Pool(processes=pool_size) as p:
        #     p.map(given_ij, [(i,j) for i in range(n_sig) for j in range(num_chains)])
    df = pd.DataFrame(np.stack([
        r2.flatten(), rmax.flatten(), chain_id.flatten(), kuhns.flatten()
        ], axis=1), columns=['r2', 'rmax', 'chain_id', 'kuhn'])
    df['mu'] = mu
    return df

def tabulate_r2_heterogenous_fluctuating_chains_homogenous(num_chains, chain_length, mu=35, pool_size=None, **kwargs):
    """Tabulate R^2 for fluctuating heterogenous chains with increasing variance. Pass unwrapping parameters
    through kwargs."""
    links = np.zeros((num_chains, chain_length-1))
    #For now, assume the same unwrapping amounts for all chains
    #w_ins, w_outs = convert.resolve_wrapping_params(unwraps, w_ins, w_outs, chain_length)
    links = mu*np.ones((num_chains,chain_length-1))
    rmax = np.zeros((num_chains, chain_length))
    r2 = rmax.copy()
    chain_id = rmax.copy()
    kuhns = rmax.copy()
    def given_chain_i(i):
        #note r2, ldna returned by R2_kinked_WLC() is the same shape as links. add a 0 to the beginning
        #to match bruno's code; Output is in nm
        R2, Rmax, kuhn = R2_kinked_WLC_no_translation(links[i,:].flatten(), plotfig=False, **kwargs)
        r2[i] = np.concatenate(([0], R2))
        rmax[i] = np.concatenate(([0], Rmax))
        kuhns[i] = kuhn
        chain_id[i] = i
    if pool_size is None:
        for i in range(num_chains):
            given_chain_i(i)
    else:
        raise NotImplementedError('No Pool here plz.')
        # with Pool(processes=pool_size) as p:
        #     p.map(given_ij, [(i,j) for i in range(n_sig) for j in range(num_chains)])
    df = pd.DataFrame(np.stack([
        r2.flatten(), rmax.flatten(), chain_id.flatten(), kuhns.flatten()
        ], axis=1), columns=['r2', 'rmax', 'chain_id', 'kuhn'])
    df['mu'] = mu
    return df

def plot_heterogenous_chain_r2(df, mu, ax=None, running_avg=True, **kwargs):
    """Read in tabulated R^2 of heterogenous chains
    generated by ncg.tabulate_r2_heterogenous_chains_by_variance(), and plot
    :math:`\langle{R^2}\rangle` vs. :math:`R_{max}`.

    Columns = ['r2', 'rmax', 'variance', 'chain_id']
    """

    if ax is None:
        fig, ax = plt.subplots()

    df2 = df.sort_values('rmax')
    window_size = 100 #window_size for rolling average of rigid rod heterogenous chains
    def newdf(df):
        return df.rolling(100).mean()
    if running_avg:
        df2 = df2.groupby('variance').apply(newdf)
    sns.lineplot(data=df2, x='rmax', y='r2', hue='variance', ci=None, ax=ax, **kwargs)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'$R_{max} (nm)$')
    plt.ylabel(r'$\langle{R^2}\rangle (nm^2)$')
    plt.title(f'$\langle{R^2}\rangle$ for heterogenous chain, $\mu={mu}$, $\sigma=0-10bp$')

def plot_r2_fluctuating_vs_geometry(dff, dfg, mu, ax=None, running_avg=True, **kwargs):
    """Plot :math:`\langle{R^2}\rangle` vs. :math:`R_{max}` for fluctuating
    chain vs. geometrical chain.

    Columns = ['r2', 'rmax', 'variance', 'chain_id']
    """

    if ax is None:
        fig, ax = plt.subplots()

    #First plot geometrical case 'dfg'
    #dfg['rmax'] = dfg['rmax']*ncg.dna_params['lpb'] #convert rmax from bp to nm
    df2g = dfg.sort_values('rmax')
    window_size = 100 #window_size for rolling average of rigid rod heterogenous chains
    def newdf(df):
        return df.rolling(100, min_periods=1).mean()
    if running_avg:
        df2g = df2g.groupby('variance').apply(newdf)
    num_colors = len(np.unique(df2g.variance))
    violets = sns.cubehelix_palette(num_colors)
    #df2g.plot(x='rmax', y='r2', label='geometrical', colors=colors)
    sns.lineplot(data=df2g, x='rmax', y='r2', hue='variance', legend='full', palette=violets, ci=None, ax=ax)

    #Next plot fluctuating case 'dff' --- no running average
    df2f = dff.sort_values('rmax')
    num_colors = len(np.unique(df2f.variance))
    greens = sns.cubehelix_palette(num_colors, start=2)
    #df2f.plot(x='rmax', y='r2', label='fluctuations', colors=colors)
    if running_avg:
        df2f = df2f.groupby('variance').apply(newdf)
    sns.lineplot(data=df2f, x='rmax', y='r2', hue='variance', legend='full', palette=greens, ci=None, ax=ax)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'$R_{max} (nm)$')
    plt.ylabel(r'$\langle{R^2}\rangle (nm^2)$')
    #plt.legend()

def get_kuhn(df, thresh, rmax_col='rmax', r2_col='r2'):
    """Take a df with r2/rmax columns and a threshold (burn in length) in
    number of monomers after which to fit adn do a linear fit to extract teh
    kuhn length."""
    ks = scipy.stats.linregress(df[rmax_col].iloc[thresh:], df[r2_col].iloc[thresh:])
    return ks

def get_kuhns_grouped(df, thresh, groups, rmax_col='rmax', r2_col='r2'):
    ks = df.groupby(groups)[[rmax_col, r2_col]].apply(get_kuhn,
            thresh=thresh, rmax_col=rmax_col, r2_col=r2_col)
    ks = ks.apply(pd.Series)
    ks.columns = ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']
    ks['b'] = ks['slope']
    return ks

def aggregate_existing_kuhns(glob='*.csv'):
    """WARNING: not really for direct use. an example function that appends new
    r2 calculations to existing repository of kuhn lengths.

    you should be able to modify for further use by just adding match_*
    variables that correspond to your format string in the r2-tabulation.py
    script"""
    kuhns = []
    for path in Path('./csvs/r2').glob(glob):
        try:
            df = pd.read_csv(path)
        except:
            continue
        match_npy = re.search('kuhns-(fluctuations|geometrical)-mu([0-9]+)-sigma_([0-9]+)_([0-9]+)_([0-9]+)unwraps.npy', str(path))
        match_box = re.search('r2-(fluctuations|geometrical)-mu_([0-9]+)-sigma_([0-9]+)_?([0-9]+)?_([0-9]+)unwraps(?:-[0-9]+)?(-random_phi)?.csv', str(path))
        match_new = re.search('r2-(fluctuations|geometrical)-(box|exponential|homogenous)-link-mu_([0-9]+)-(?:sigma_([0-9]+))?_?([0-9]+)unwraps(?:-[0-9]+)?(-random_phi)?.csv', str(path))
        match_random_phi_exp = re.search('r2-(fluctuations|geometrical)-mu_([0-9]+)-sigma_([0-9]+)_([0-9]+)unwraps_random-phi-rz-(left|right).csv', str(path))
        if match_box is not None:
            variance_type = 'box'
            sim, mu, sigma_min, sigma_max, unwraps, is_random = match_box.groups()
            if is_random:
                sim = sim + '-random_phi'
            groups = ['mu', 'variance']
        elif match_new is not None:
            sim, variance_type, mu, sigma, unwraps, is_random = match_new.groups()
            if variance_type == 'homogenous' and sigma is not None:
                variance_type = 'box'
            if is_random:
                sim = sim + '-random_phi'
            groups = ['mu']
        elif match_npy is not None:
            variance_type = 'box'
            sim, mu, sigma_min, sigma_max, unwraps = match_npy.groups()
        elif match_random_phi_exp is not None:
            variance_type = 'box'
            sim1, mu, sigma, unwraps, sim2 = match_random_phi_exp.groups()
            # it was later discovered that left application is the correct one
            if sim2 == 'right':
                continue
            sim = f'{sim1}-random_phi'
        else:
            print('Unknown simulation type: ' + str(path))
            continue
        df['mu'] = mu
        ks = get_kuhns_grouped(df, thresh=5000, groups=groups)
        ks = ks.reset_index()
        ks['type'] = sim
        ks['variance_type'] = variance_type
        ks['unwrap'] = unwraps
        if 'homogenous' == ks['variance_type'].iloc[0]:
            ks['variance'] = 0
        elif 'exponential' == ks['variance_type'].iloc[0]:
            ks['variance'] = ks['mu']
        elif match_new and sigma is not None:
            ks['variance'] = sigma
        else:
            assert('variance' in ks)
        kuhns.append(ks)
    all_ks = [ks.set_index(['variance_type', 'type', 'mu', 'variance', 'unwrap']) for ks in kuhns]
    all_ks = pd.concat(all_ks)

    # df = pd.read_csv('csvs/r2/kuhn_lengths_so_far.csv')
    # df['variance_type'] = 'box'
    # all_ks['variance'] = all_ks['mu']
    # all_ks['variance_type'] = 'exponential'
    # df.set_index(['variance_type', 'type', 'mu', 'variance'], inplace=True)
    # all_ks.set_index(['variance_type', 'type', 'mu', 'variance'], inplace=True)
    # all_ks.sort_index(inplace=True)
    # all_ks = pd.concat([ak, df])
    # all_ks.to_csv('./csvs/r2/kuhn_lengths_so_far.csv')

    return all_ks

def calculate_kuhn_length_from_fluctuating_r2(df, mu, chain_length, **kwargs):
    """Calculate :math:`b=\langle{R^2}\rangle/R_{max}` in the long chain
    limit (roughly 5000 monomers down the chain) by averaging kuhn lengths
    from each individual chain."""

    df2 = df.sort_values('rmax')
    kuhns = []
    for var, vals in df2.groupby(['variance']):
        #take the average kuhn length of each individual chain in this variance group
        kuhns.append(np.mean(vals['kuhn']))
    return np.array(kuhns)

def calculate_kuhn_length_from_r2(df, mu, chain_length, **kwargs):
    """Calculate :math:`b=\langle{R^2}\rangle/R_{max}` in the long chain
    limit (roughly 5000 monomers down the chain). Sorts df, finds threshold corresponding
    roughly to 5000th monomer and does linear fit beyond this threshold."""

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

def plot_kuhn_length_vs_variance(kuhnsf, kuhnsg, mu, sigmas=np.arange(0, 11), ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(sigmas, kuhnsg, '--o', label='T=0')
    ax.plot(sigmas, kuhnsf, '-o', label='Fluctuations')
    plt.xlabel(r'Variance in $\phi$')
    plt.ylabel('Kuhn Length (nm)')
    plt.title(f'Rigid Rod vs. Fluctuating Chain, mu={mu}bp')
    plt.ylim([0, 200])
    plt.legend()

def visualize_kuhn_lengths(links, unwraps, kuhns, mfig=None, **kwargs):
    """Plot 3D surface of Kuhn length as a function of constant linker length, unwrapping amount."""

    if mfig is None:
        mfig = mlab.figure()

    mlab.surf(links, unwraps, kuhns)
    mlab.axes()
    mlab.xlabel('Link (bp)')
    mlab.ylabel('Unwrap (bp)')
    mlab.zlabel('Kuhn length (nm)')
###}}}

###{{{
# """Propogators and Greens Functions"""

def tabulate_M_kinks(unwraps=None, l0max=None, **kwargs):
    """Return a lookup table of M matrices for a given (alpha, beta, gamma).
    One matrix for each possible level of unwrapping.

    Returns
    -------
    Mdict : dictionary
        key = unwrapping amount in bp, values = 441 by 441 matrix

    Saves
    -----
    Mkink_matrices.csv : file from pd.DataFrame
        indexed by unwrapping amount --- each matrix is 441 by 441
    Mdict_from_unwraps.p : pickle dumped dictionary
        to load, Mdict = pickle.load(open('Mdict_from_unwraps.p', 'rb'))
    """
    #retrieve dictionary of 441 by 441 M kink matrices if it exists
    mdicts_file = ncd.data_dir / Path('Mdict_from_unwraps.p')
    if l0max is None and unwraps is None and mdicts_file.exists():
        return pickle.load(open(mdicts_file, 'rb'))

    if l0max is None:
        l0max = 20
    if unwraps is None:
        unwraps = np.arange(0, 147)

    tau_n = kwargs['tau_n'] if 'tau_n' in kwargs else ncg.dna_params['tau_n']
    # super indexing
    ntot = l0max**2 + 2*l0max + 1
    I  = lambda l, j: l**2 + l + j

    # create dictionary of M's from unwrapping amount to super-indexed M matrix
    Mdict = {}
    mywd = wd.wigner_d_vals()

    for i, u in enumerate(unwraps):
        w_in, w_out = convert.resolve_unwrap(u)
        R = ncg.OmegaE2E(w_in+w_out, tau_n=tau_n)
        alpha, beta, gamma = ncr.zyz_from_matrix(R)

        M = np.zeros((ntot, ntot), 'complex')
        # Construct super-indexed M matrix for kink defined by this unwrapping amount
        for l0 in range(l0max+1):
            for j0 in range(-l0, l0+1):
                I0 = I(l0, j0)
                for jf in range(-l0, l0+1):
                    If = I(l0, jf)
                    M[If, I0] = mywd.get(l0, jf, j0, -gamma, -beta, -alpha) / mywd.normalize(l0, jf, j0)
        # Mdict[u] = pd.DataFrame(M)
        Mdict[u] = M

    #CODE USED TO SAVE CSV FILE -- module uses pickled file for calculations
    # Mdf = pd.concat(Mdict.values(), keys=Mdict.keys())
    # Mdf.index.names = ['unwrap', 'If']
    # Mdf.to_csv('csvs/Mkink_matrices.csv')
    pickle.dump(Mdict, open(mdicts_file, 'wb'))
    return Mdict

Mdict = tabulate_M_kinks()
"""Pickled files for Greens function calculations"""


###{{{
# """Linker propogator g calculations"""

def build_A_matrix(j, k, lmax, **kwargs):
    """Build sparse A matrix in the ode for g propogators: dg/dL = Ag.
    Dimension of matrix is (lmax + 1) by (lmax + 1).

    Helper function for gprop_k_given_link."""

    # make sure A is at least 20 by 20 in size (otherwise matrix is trivial)
    if (lmax - abs(j)) < 20:
        raise ValueError('lmax must be at least abs(j) + 20')

    lp = kwargs['lp'] if 'lp' in kwargs else default_lp
    lt = kwargs['lt'] if 'lt' in kwargs else default_lt
    tau_d = kwargs['tau_d'] if 'tau_d' in kwargs else ncg.dna_params['tau_d']

    al  = lambda l, j: np.sqrt((l-j)*(l+j)/(4*l**2 - 1)) # ladder coefficients alpha
    lam = lambda l, j: (l*(l+1))/(2*lp) + 0.5*((1/lt)-(1/lp))*j**2 - 1j*tau_d*j # eigenvalue of H0

    # construct diagonals of tridiagonal matrix
    lowerdiag = [1j*k*al(l, j) for l in range(abs(j)+1, lmax+1)]
    maindiag  = [-lam(l, j) for l in range(abs(j), lmax+1)]
    upperdiag = lowerdiag

    # was getting sparse efficiency warnings from scipy unless I specified 'csc' format
    A = sparse.diags([lowerdiag, maindiag, upperdiag], [-1, 0, 1], format='csc')
    return A

def gprop_k_given_link(k, link, unwrap, l0max=20, lmax=None, **kwargs):
    """Solve dg/dL = Ag, where A is a tridiagonal matrix.

    Notes
    -----
    This function takes roughly 3s to run on a new (k, link, unwrap), and ~183ms to run on
    a pre-computed tuple.
    """

    # set some arbitrarily large limit on lf -- so A is at most a 50x50 matrix, at leas ta 20x20 matrix
    if lmax is None:
        lmax = l0max + 30

    #Creating this path object + checking if it's a file = 14.3 us
    gfile = Path(f'csvs/gprops/{unwrap:.0f}unwraps/{link:.0f}link/gprop_k{k}_{link:.0f}bplinks.csv')
    try:
        g = parse_gprop_csv_file(k, link, unwrap, l0max)
        return g
    except:
        if gfile.is_file():
            gfile.unlink()
        print(f"Failed to parse csv file for k = {k}, unwrap = {unwrap}, link = {link}")
        print(f"Recomputing...")


    # otherwise calculate g propagator, save to csv

    #check if all parent directories exist before writing to above file
    try:
        gfile.parent.mkdir(parents=True)
    except:
        if gfile.parent.is_dir() is False:
            #if the parent directory does not exist and mkdir still failed, re-raise an exception
            raise
    #total linker length
    Ll = link + unwrap
    # construct super indexed matrix g(I0, If)
    ntot = l0max**2 + 2*l0max + 1
    I   = lambda l, j: l**2 + l + j
    g = np.zeros((ntot, ntot), 'complex')
    #save list of Pandas DataFrames to be concatenated into csv
    gsols = []
    for j0 in range(-l0max, l0max+1):
        # ODE: dg/dL = Ag
        A = Ll*build_A_matrix(j0, k, lmax, **kwargs)
        # initial condition: g(L=0): all 0's except when l = l0
        # gsol contains the exponentiated matrix where rows are l and columns are l0

        # I get a scipy sparse efficiency warning when performing the matrix exponentiation
        # it recommends the 'lil' format, but when I tried that, I got different sparse efficiency
        # warnings. Going to stick with 'csc' for now.
        gsol  = sparse.linalg.expm(A)
        for l0 in range(abs(j0), l0max+1):
            I0 = I(l0, j0)
            for l in range(abs(j0), l0max+1):
                If = I(l, j0)
                g[If, I0] = gsol[l-abs(j0), l0-abs(j0)]
        df = pd.DataFrame(gsol[0:(l0max+1-abs(j0)),
                                0:(l0max+1-abs(j0))].toarray())
        df['k'] = k
        df['link'] = link
        df['j0'] = j0
        gsols.append(df)

    gdf = pd.concat(gsols, ignore_index=True, sort=False)
    gdf.set_index(['k', 'link', 'j0'], inplace=True)
    #add a file lock so that multiple processes cannot write to the same file at once
    lock = FileLock(str(gfile) + '.lock')
    with lock.acquire():
        gdf.to_csv(gfile)
    print(f'saved gprop for k={k}, link={link}bp, unwrap={unwrap}bp')
    sys.stdout.flush()
    return g

def save_gprop_csv_file(k, link, unwrap, l0max=20, lmax=None, **kwargs):
    """Saves solutions to dg/dL = Ag ODEs to a csv file, does not return anything.
    Use this function for tabulating linker propagators when the
    super-indexed matrix is not going to be used immediately."""

    # set some arbitrarily large limit on lf -- so A is at most a 50x50 matrix, at least a 20x20 matrix
    if lmax is None:
        lmax = l0max + 30

    #Path of file to be created
    gfile = Path(f'csvs/gprops/{unwrap:.0f}unwraps/{link:.0f}link/gprop_k{k}_{link:.0f}bplinks.csv')

    if (gfile.is_file() is False):
        #check if all parent directories exist before writing/reading above file:
        #missing parents of path are created as needed, FileExists error will be thrown if attempt
        #to make a directory that already exists
        try:
            gfile.parent.mkdir(parents=True)
        except:
            # might fail because the directory was created by someone else
            if gfile.parent.is_dir() is False:
                #if the parent directory does not exist and mkdir still failed, re-raise an exception
                raise
        #calculate g propagator, save to csv
        #total linker length
        Ll = link + unwrap
        #save list of Pandas DataFrames to be concatenated into csv
        gsols = []
        for j0 in range(-l0max, l0max+1):
            # ODE: dg/dL = Ag
            A = Ll*build_A_matrix(j0, k, lmax, **kwargs)
            # initial condition: g(L=0): all 0's except when l = l0
            # gsol contains the exponentiated matrix where rows are l and columns are l0
            # I get a scipy sparse efficiency warning when performing the matrix exponentiation
            # it recommends the 'lil' format, but when I tried that, I got different sparse efficiency
            # warnings. Going to stick with 'csc' for now.
            gsol  = sparse.linalg.expm(A)
            df = pd.DataFrame(gsol[0:(l0max+1-abs(j0)),
                                   0:(l0max+1-abs(j0))].toarray())
            df['k'] = k
            df['link'] = link
            df['j0'] = j0
            gsols.append(df)

        gdf = pd.concat(gsols, ignore_index=True, sort=False)
        gdf.set_index(['k', 'link', 'j0'], inplace=True)
        gdf.to_csv(gfile)
        print(f'k={k}, link={link}bp, unwrap={unwrap}bp')

def parse_gprop_csv_file(k, link, unwrap, l0max=20):
    """Reads in csv file containing saved g propagators. Assumes the columns of the csv are
    k, link, j0, 0-l0max+1. Parses complex matrix values and reorganizes data into
    super-indexed g[If, I0] where :math:`I(l, j) = l^2 + l + j`. As always,
    :math:`I` can take on :math:`l^2+2l+1` possible values."""

    #### READ CSV: 15.3ms #####
    inds = [i for i in range(l0max+1)]

    #for 36bp linkers, file format was different. Columns are 'k', 'link', 'j0', 'index', [inds]
    if (link == 36):
        df = pd.read_csv(f'csvs/gprops/{unwrap:.0f}unwraps/{link:.0f}link/gprop_k{k}_{link:.0f}bplinks.csv',
                        header=0, names=['k', 'link', 'j0', 'ind']+inds)
        #effectively remove the random index column in between 'j0' and the matrix values
        df = df[['k', 'link', 'j0']+inds]
    else :
        df = pd.read_csv(f'csvs/gprops/{unwrap:.0f}unwraps/{link:.0f}link/gprop_k{k}_{link:.0f}bplinks.csv',
                        header=0, names=['k', 'link', 'j0']+inds)

    #### CONVERT COMPLEX: 37.9 ms ######
    def complexify(x):
        try:
            return np.complex(x)
        except ValueError as ve:
            print(f'Failed to convert {x} into a complex number.')
            sys.stdout.flush()
            raise ve
    for i in inds:
        df[i] = df[i].str.replace(' ', '') #remove extraneous white spaces
        df[i] = df[i].str.replace('(', '')
        df[i] = df[i].str.replace(')', '')
        df[i] = df[i].str.replace('i','j').apply(complexify)

    #### SUPERINDEX : 100 ms ####
    # construct super indexed matrix g(I0, If)
    ntot = l0max**2 + 2*l0max + 1
    g = np.zeros((ntot, ntot), 'complex')
    I   = lambda l, j: l**2 + l + j

    j0 = -l0max
    for key, matrix in df.groupby(['k','link','j0']):
        for l0 in range(abs(j0), l0max+1):
            I0 = I(l0, j0)
            for l in range(abs(j0), l0max+1):
                If = I(l, j0)
                # +3 to skip 'k', 'link', 'j0' columns
                g[If, I0] = matrix.iat[l-abs(j0), l0-abs(j0)+3]
        j0 += 1
    return g

###}}}

###{{{
# """Bare WLC Calculations"""
# various functions renormalized to match Quinn/Shifan's WLC code to check that
# ours works

def gprop_K_given_N(K, N, l0max=20, **kwargs):
    """Solve for the linker propogator without twist to compare to Quinn's code.
    Non-dminesionalize k and linker length to obtain G(K;N) where K=2lpk, N=L/(2lp).
    """

    # set persistance length to be super high (effectively no twist)
    lt = kwargs['lt'] if 'lt' in kwargs else 10000*default_lt
    lp = kwargs['lp'] if 'lp' in kwargs else default_lp
    tau_d = kwargs['tau_d'] if 'tau_d' in kwargs else ncg.dna_params['tau_d']

    link = (2*lp)*N
    k = K/(2*lp)

    return gprop_k_given_link(k, link, l0max, lp=lp, lt=lt, tau_d=tau_d)

def get_G(K, N, l, l0, j, l0max=20, lmax=None, **kwargs):
    """Return :math:`g_{l l_0}^j(K;N)`. Only need to solve one ODE for given value of j."""

    if (abs(j) > l) or (abs(j) > l0):
        raise ValueError('abs(j) must be less than or equal to both l and l0')

    if lmax is None:
        lmax = l0max + 30

    # set twist persistance length to be super high (effectively no twist)
    lt = kwargs['lt'] if 'lt' in kwargs else 10000*default_lt
    lp = kwargs['lp'] if 'lp' in kwargs else default_lp
    tau_d = kwargs['tau_d'] if 'tau_d' in kwargs else ncg.dna_params['tau_d']

    link = (2*lp)*N
    k = K/(2*lp)

    A = link*build_A_matrix(j, k, lmax, lp=lp, lt=lt, tau_d=tau_d)
    gsol = sparse.linalg.expm(A)
    return gsol[l-abs(j), l0-abs(j)]

def plot_GKN(Ks, Ns, l, l0, j):
    """Plotting code for G(K;N) vs K for a given l, l0, j. Plots one curve for each N."""
    fig, ax = plt.subplots()
    Gs = np.zeros((Ns.size, Ks.size), 'complex')
    for nn in range(Ns.size):
        for kk in range(Ks.size):
            Gs[nn, kk] = get_G(Ks[kk], Ns[nn], l, l0, j)
        ax.loglog(Ks, np.abs(Gs[nn, :].real))

    plt.xlabel('K')
    plt.ylabel('G(K;N)')
    plt.title(f'$l={l}, l_0={l0}, j={j}$')
    plt.legend([f'N={N}' for N in Ns])
    plt.ylim([10**-12, 2])
    plt.show()
    return Gs

def GRN_fourier_integrand(K, r, N):
    """Un-normalized fourier-inversion integrand for bare WLC."""
    G0 = get_G(K, N, 0, 0, 0)
    return K**2 * special.spherical_jn(0, N*r*K) * G0.real

def gprop_R_given_N_quad_integration(Kmin=10**-3, Kmax=10**5, l0max=20, **kwargs):
    """Return the un-normalized, real-space Green's function G(R;N) for bare WLC,
    using scipy's quad integration and GRN_fourier_integrand(K, r, N)
    Performs integration across 5 orders of magnitude for N and rvals from 0 to 1,
    where r=R/L, N=L/(2lp)."""

    rvals = np.linspace(0.0, 1.0, 100) # R/L
    Nvals = np.array([0.1, 1.0, 10.0, 100., 1000.])
    integral = np.zeros((rvals.size, Nvals.size))
    errs = np.zeros_like(integral)

    for i, r in enumerate(rvals):
        for j, N in enumerate(Nvals):
            sol, err = quad(GRN_fourier_integrand, Kmin, Kmax, args=(r, N))
            integral[i, j] = sol
            errs[i, j] = err
        pickle.dump(integral, open(f'GRN_integral_K{Kmin}to{Kmax}_r0to1.p', 'wb'))
        pickle.dump(errs, open(f'GRN_errors_K{Kmin}to{Kmax}_r0to1.p', 'wb'))
        print(f'Computed integral for r={r}')

    np.save(f'GRN_integral_K{Kmin}to{Kmax}_r0to1', integral)
    np.save(f'GRN_errors_K{Kmin}to{Kmax}_r0to1', errs)
    return integral, errs
###}}}

###{{{
# """Propogators for kinked WLC"""

#@profile
def Bprop_k_given_L(k, links, filepath, w_ins=ncg.default_w_in, w_outs=ncg.default_w_out,
                        helix_params=ncg.helix_params_best, unwraps=None, **kwargs):
    """Calculate :math:`B_{00}^{00}(k;L)` for a chain of heterogenous linkers.
    NOTE: this is the only function for heterogenous chains that allows for different
    unwrapping amounts. All other functions in the pipeline (aka Fourier inversion,
    looping, etc.) assume fixed unwrapping amount. These other functions need to be
    modified to accept variable unwrapping.

    Parameters
    ----------
    k : float
        k value for which B(k;L) should be calculated, in :math:`bp^{-1}`
    links : (L,) array-like
        bare linker lengths in bp
    filepath : str or pathlib.Path object
        full path to folder where output should be saved,
        should include name to identify this particular heterogenous chain
        e.g.: 'csvs/Bprops/0unwraps/heterogenous/{chain_identifier}'
    w_ins : float or (L+1,) array_like
        amount of DNA wrapped on entry side of central dyad base in bp
    w_outs : float or (L+1,) array_like
        amount of DNA wrapped on exit side of central dyad base in bp

    Notes
    -----
    Adding a new propagator to a chain takes roughly 200ms.

    Saves
    -----
    'Bprop_k{k}_given_L_{len(links)}nucs.npy' : binary file
        :math:`B_{000}^{000}(k; L)`, one per monomer in chain
        Saved in 'csvs/Bprops/0unwraps/heterogenous/chain_identifier'
    """

    #save this singleton array in npy format since pool isn't letting me save at the end
    #check if all parent directories exist before writing/reading file:
    filepath = Path(filepath) #in case str is passed
    Bfile = filepath/Path(f'Bprop_k{k}_given_L_{len(links)}nucs.npy')

    #if file already exist, don't bother recalculating. Move on to next k.
    try:
        np.load(Bfile)
        return
    except:
        # if there's some error loading the Bfile, delete it if it exists
        if Bfile.is_file():
            Bfile.unlink()

    b = helix_params['b']
    num_linkers = len(links)
    num_nucleosomes = num_linkers + 1
    w_ins, w_outs = convert.resolve_wrapping_params(unwraps, w_ins, w_outs, num_nucleosomes)
    # calculate unwrapping amounts based on w_ins and w_outs
    mu_ins = (b - 1)/2 - w_ins
    mu_outs = (b - 1)/2 - w_outs

    # save array of propogators of size (num_linkers,)
    Bprops = np.zeros((num_linkers,)).astype('complex')
    #save dictionary of B matrices in case there are a lot of repeat linkers
    Bmats = {}

    #first link in chain
    unwrap = mu_outs[0] + mu_ins[1]
    link = links[0]
    try:
        g = gprop_k_given_link(k, link, unwrap, **kwargs)
    except Exception as e:
        print('ERROR: k={k}, link={link}, unwrap={unwrap}')
        sys.stdout.flush()
        raise e
    M = Mdict[unwrap]
    Bcurr = M@g
    Bmats[(link, unwrap)] = Bcurr
    Bprops[0] = Bcurr[0, 0]

    for i in range(1, num_linkers):
        unwrap = mu_outs[i] + mu_ins[i+1]
        link = links[i]
        key = (link, unwrap)
        if key not in Bmats:
            # our M kink matrices are index by *unwrapping* amount, not wrapped amount
            # calculate propagators
            g = gprop_k_given_link(k, link, unwrap, **kwargs)
            M = Mdict[unwrap]
            Bnext = M@g
            Bmats[key] = Bnext
        Bnext = Bmats[key]
        # advance chain by one linker
        Bcurr = Bnext@Bcurr
        Bprops[i] = Bcurr[0, 0]

    try:
        Bfile.parent.mkdir(parents=True)
    except:
        if Bfile.parent.is_dir() is False:
            #if the parent directory does not exist and mkdir still failed, re-raise an exception
            raise

    #save file
    np.save(Bfile, Bprops, allow_pickle=False)
    print(f'wrote bprop file for k={k}, link={links[0]}')
    sys.stdout.flush()

def combine_Bprop_files(filepath, links, unwraps, kvals=None, bareWLC=True,  **kwargs):
    """Combine each of the files saved by Bprop_k_given_L() into a single matrix,
    compute the Fourier inversion for the kinked WLC and corresponding bare WLC of
    the same length and save them all in the directory specified by 'filepath'.

    Parameters
    ----------
    filepath : Path object or str
        full path to directory where files from Bprop_k_given_L() were saved
        e.g. 'csvs/Bprops/0unwraps/heterogenous/chain_identifier'
    links : (L,) array-like
        bare linker lengths in bp, one for each monomer in heterogenous chain
    unwraps : int
        unwrapping amount in bp. Assumes fixed unwrapping.
    """

    num_linkers = len(links)
    filepath = Path(filepath) #in case str is passed

    if kvals is None:
        Klin = np.linspace(0, 10**5, 20000)
        Klog = np.logspace(-3, 5, 10000)
        Kvals = np.unique(np.concatenate((Klin, Klog)))
        #convert to little k -- units of inverse bp (this results in kmax = 332)
        kvals = Kvals / (2*default_lp)

    Bprops = np.zeros((len(kvals), num_linkers)).astype('complex')
    for kk, k in enumerate(kvals):
        Bfile = filepath/f'Bprop_k{k}_given_L_{len(links)}nucs.npy'
        #try loading in Bprop file; if faulty, halt combine process
        try:
            Bprops[kk, :] = np.load(Bfile)
        except:
            # if there's some error loading the Bfile, delete it if it exists
            if Bfile.is_file():
                print(f'Problem reading file {Bfile}... deleting...')
                Bfile.unlink()
            print(f'Recomputing bprop for k={k}...')
            Bprop_k_given_L(k, links, filepath)
            try:
                Bprops[kk, :] = np.load(Bfile)
            except:
                print('Recomputed file also can not be loaded. Halting combine process.')
                raise

    chain_identifier = filepath.name #parent directory of Bprop files = chain_identifier
    np.save(filepath/Path(f'linker_lengths_{chain_identifier}_{num_linkers}nucs.npy'),
            links, allow_pickle=False)
    np.save(filepath/Path(f'B0_k_given_L_{chain_identifier}_{num_linkers}nucs_30000Ks.npy'), Bprops, allow_pickle=False)
    if bareWLC:
        qprop = bareWLC_gprop(kvals, links, unwraps, **kwargs)
        qintegral = BRN_fourier_integrand_splines(kvals, links, unwraps, Bprop=qprop, **kwargs) #default: 1000 rvals
        np.save(filepath/Path(f'bareWLC_greens_{chain_identifier}_{num_linkers}nucs.npy'), qintegral, allow_pickle=False)
    integral = BRN_fourier_integrand_splines(kvals, links, unwraps, Bprop=Bprops, **kwargs) #default: 1000 rvals
    np.save(filepath/Path(f'kinkedWLC_greens_{chain_identifier}_{num_linkers}nucs.npy'), integral, allow_pickle=False)
    print(f'Saved G(R;L) for {chain_identifier}, {num_linkers} monomers!')
    sys.stdout.flush()

def Bprop_k_given_L_Sarah_chain_Rlinks(k, unwrap=0, Nvals=None, **kwargs):
    """Grow a chain that has a finite set of heteoregenous linkers followed by constant linkers. The first part
    will be calculated by manually multiplying propagators, and the second will use matrix exponentiation."""

    #for Rlinks, I actually saved the matrices obtained from 7 links: 47, 21, 18, 15, 20, 17, 35
    #so just need to caculate propagator for 35 link and exponentiate to get rest of chain
    Bcurr = pickle.load(open(f'csvs/Bprops/0unwraps/Sarah/Bmatrices_Rlinks_7nucs/Bmatrix_k{k}_Sarah_Rlinks_7nucs_no_unwraps.p', 'rb'))

    #for constant 35bp linkers
    links = np.tile(35, 493) #so total chain is 500 nucs
    num_linkers = len(links)

    if Nvals is None:
        Nvals = np.arange(1, num_linkers+1)

    Bprops = np.zeros_like(Nvals).astype('complex')
    #B propagator for 35 bp link, 0 unwrap
    g = gprop_k_given_link(k, links[0], unwrap, **kwargs)
    M = Mdict[unwrap]
    B = M@g

    for i, N in enumerate(Nvals):
        #exponentiate propagator
        BN = np.linalg.matrix_power(B, N)
        #multiply by heterogenous propagator from first 7 nucs in chain
        Bfinal = BN@Bcurr
        Bprops[i] = Bfinal[0, 0]

    print(f'Rlinks, 450to500nucs, k=({k}')
    #this only includes linkers 8-50; properly append it to pre-saved Bprops for the first 7 nucs
    return Bprops


def Bprop_k_given_link(k, link, unwrap, **kwargs):
    """Calculates B propagator for a single monomer with linker length 'link' and unwrapping
    amount 'unwrap'.

    Parameters
    ----------
    k : float
        k value for which to compute B(k; L) in :math:`bp^{-1}`
    link : int
        bare linker length in bp (not including unwrapping)
    unwrap : int
        unwrapping amount in bp

    Returns
    -------
    B00 : complex
        :math:`B_{00}^{00}(k;N=1)` component of propagator
    """
    #total linker length is bare linker + unwrapping amount
    g = gprop_k_given_link(k, link, unwrap, **kwargs)
    M = Mdict[unwrap]
    B = M@g
    return B[0, 0]


@utils.cache
def Bprop_k_given_L_fixed_linkers_fixed_unwrap(k, N, link, unwrap, **kwargs):
    """Calculate propagator for Nth monomer in a chain with fixed linkers
    and fixed unwrapping amounts.

    Parameters
    ----------
    k : float
        k value for which to compute B(k; L) in :math:`bp^{-1}`
    N : int
        number of nucleosomes down the chain (i.e. Nth propogator)
    link : int
        bare linker length in bp (not including unwrapping)
    unwrap : float
        unwrapping amount in bp

    Returns
    -------
    B00 : complex
        :math:`B_{00}^{00}(k;N)` component of Nth propagator
    """
    g = gprop_k_given_link(k, link, unwrap, **kwargs)
    M = Mdict[unwrap]
    B = M@g
    BN = np.linalg.matrix_power(B, N)
    return BN[0, 0]

def Bprop_k_fixed_linkers_fixed_unwrap(k, links, unwrap, Nvals=None, **kwargs):
    """Same as Bprop_k_given_L except assumes fixed linker length and constant unwrapping amount.
    Under these assumptions, the B propogator for each monomer is identical, so growing the chain
    simply requires a matrix exponentiation. TODO: change this code to use for loop; matrix
    exponentiation is slow for large N.

    Parameters
    ----------
    k : float
        k value for which to compute B(k; L) in :math:`bp^{-1}`
    links : (L,) array-like
        bare linker length in bp (not including unwrapping)
    unwraps : float
        unwrapping amount in bp
    Nvals : array-like
        number of linkers down the chain for which you want the propagator. Defaults to one propagator
        per monomer of the chain.

    Returns
    -------
    Bprops : (Nvals.size,), array-like
        :math:`B_{00}^{00}(k;N)` for all N values

    """
    num_linkers = len(links)
    if(np.array_equal(np.tile(links[0], num_linkers), links) is False):
        raise ValueError('linker lengths must all be constant')

    if Nvals is None:
        Nvals = np.arange(1, num_linkers+1)

    g = gprop_k_given_link(k, links[0], unwrap, **kwargs)
    M = Mdict[unwrap]
    B = M@g
    Bprops = np.zeros_like(Nvals).astype('complex')

    #matrix exponentiation is slow for large N. Switch to for loop
    for i, N in enumerate(Nvals):
        BN = np.linalg.matrix_power(B, N)
        Bprops[i] = BN[0, 0]

    print(f'k=({k}, link={links[0]}bp, unwrap={unwrap}bp')
    return Bprops


def BRN_fourier_integrand_splines(links, unwrap, Nvals=None, rvals=None, Bprop=None):
    """Return normalized, real space Green's function G(R;Rmax) for kinked WLC using
    spline integration.

    Parameters
    ----------
    kvals : (29999,) array-like
        k values for which to compute G(k; N) in :math:`bp^{-1}`
    links : (L,) array-like
        bare linker lengths in bp (not including unwrapping), one per nuc in chain. Assumes length
        of chain in len(links).
    unwrap : float
        unwrapping amount in bp. Assumes fixed unwrapping.
    Nvals : (N,) array-like
        number of linkers down the chain for which Bprop was calculated. Defaults to one
        per monomer of the chain. (Nvals should correspond to columns of Bprop)
    rvals : array-like
        dimensionless chain length R/Rmax, where Rmax is cumulative length of linkers and R is the
        length of the polymer
    Bprop : (29999, N) array-like
        :math:`B_{00}^{00}(k;N)` for each k (rows) and each chain length N (columns).
        Defaults to propagator for chain with fixed linkers and fixed unwrapping and loads in corresponding
        pickled file.

    Returns
    -------
    greens : (rvals.size, Nvals.size), array-like
        :math:`G(R;L)` for all specified chain lengths

    """
    num_linkers = len(links)
    if Bprop is None:
        Bprop =(
        pickle.load(open(f'csvs/Bprops/{unwrap}unwraps/{links[0]}link/B0_k_given_N_{links[0]}bplinkers_{unwrap}unwraps_{len(links)}nucs_{len(kvals)}Ks.p','rb')))

    if Nvals is None:
        Nvals = np.arange(1, num_linkers+1)

    if (Nvals.size != Bprop.shape[1]):
        raise ValueError('Nvals must correspond to columns of Bprop')

    if rvals is None:
        rvals = np.linspace(0.0, 1.0, 1000) #R/L

    Rmax = convert.Rmax_from_links_unwraps(links, unwraps=unwrap)
    inds = Nvals - 1
    Rmax = Rmax[inds]
    integral = np.zeros((rvals.size, Rmax.size))

    for i, r in enumerate(rvals):
        for j in range(len(Rmax)):
            R = r*Rmax[j]
            out = kvals**2 * special.spherical_jn(0, kvals*R) * Bprop[:, j].real
            integral[i, j] = 1/(2*np.pi**2) * splint(min(kvals), max(kvals), splrep(kvals, out))

    return integral

def plot_BKN(Ks, Bprop, links, unwrap, Nvals=None):
    """Plotting code for B(K;N) vs K where N is the number of nucleosomes in the chain.
    Assumes Bprop's rows are K values and columns are N values."""

    if Nvals is None:
        Nvals = np.arange(1, len(links)+1)

    fig, ax = plt.subplots()
    ldna = convert.genomic_length_from_links_unwraps(links, unwraps=unwrap)
    inds = Nvals - 1
    Ls = ldna[inds]
    for i in inds:
        ax.loglog(Ks, Bprop[:, i].real)
    plt.xlabel('K')
    plt.ylabel('B(K;L)')
    plt.title(f'$B_0^0(K;L)$ for {links[0]}bp linkers, {unwrap}bp unwrapping')
    plt.legend([f'L={L:.0f}bp' for L in Ls])
    plt.ylim([10**-12, 2])
    return fig, ax

def plot_fourier_integrand(kvals, Bprop, R, N):
    """Plotting code for the actual function being integrated to obtain G(R;N)
    Use this code and zoom in on ranges of k values to ensure K spacing is
    accurant enough to capture periodicity in integrand."""

    fig, ax = plt.subplots()
    outs = kvals**2 * special.spherical_jn(0, kvals*R) * Bprop[:, N-1].real
    ax.plot(kvals, outs, '-o')

def plot_greens(integral, links, unwrap, Nvals, rvals=None, rminN1=0.0, ax=None):
    """Plot G(R;Rmax) vs. dimensionless chain length r = R/Rmax, one curve
    per Nval, where N is the chain length in number of nucleosomes.

    Parameters
    ----------
    integral : (rvals.size, Rmax.size) array-like
        Green's function for chain with this set of linkers and unwrapping
    links : (L,) array-like
        bare linker length in bp (not including unwrapping)
    unwraps : float
        unwrapping amount in bp. Assumes fixed unwrapping.
    Nvals : array-like
        number of linkers down the chain for which you want to plot G(R;Rmax).
    rminN1 : float
        minimum r value from which the N=1 curve should be plotted. Defaults to 0.0.
        Due to numerical issues, there tends to be noise for r values < 0.7. To avoid plotting this noise,
        set rminN1=0.7 (or whatever value seems fitting for your particular chain).
    """

    if rvals is None:
        rvals = np.linspace(0.0, 1.0, 1000)

    if ax is None:
        fig, ax = plt.subplots()
    #rows are rvals, columns are N vals
    #integral = BRN_fourier_integrad_splines(kvals, links, unwrap)
    ldna = convert.genomic_length_from_links_unwraps(links, unwraps=unwrap)
    inds = Nvals - 1
    Ls = ldna[inds]

    for i in inds:
        rmin = 0.0
        if (i==0): #for N=1 case, don't plot noise -- can manually pass in the right cutoff
           rmin = rminN1
        rsub = rvals[(rvals >= rmin)]
        intsub = integral[(rvals >= rmin), i]
        ax.semilogy(rsub, intsub, '-o', markersize=2, linewidth=1)

    plt.xlabel('$R/R_{max}$')
    plt.ylabel('G(R;L)')
    plt.legend([f'L={L:.0f}bp' for L in Ls], frameon=True)
    plt.title(f'{links[0]}bp linkers, {unwrap} unwraps')
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
    fig, ax = plt.subplots()
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
###}}}

###{{{
# looping stuff

def sarah_looping(N):
    """Looping probability of a bare WLC with a capture radius of a=0.1b
    (1/10th Kuhn lengths), tabulated by Sarah. For values beyond the N that was
    tabulated, Gaussian chain behavior is assumed.

    To compare to the probabilities that we calculate in e.g. load_WLC_looping,
    just put into our units,

         >>> n, sarah_looping(n/b)/b**2

    In [1]: m, intercept, rvalue, pvalue, stderr = scipy.stats.linregres
        ...: s(np.log10(sarah_loops.n[sarah_loops.n > 10]), np.log10(sara
        ...: h_loops.pLoop[sarah_loops.n > 10]))

    In [2]: m
    Out[2]: -1.47636997617001

    In [3]: intercept
    Out[3]: -2.981924639134709
    """
    a = 0.1 # capture radius used by Sarah
    n_max = ncd.sarah_looping.n.max()
    N = np.atleast_1d(N)
    # implicitly N<0.1 ==> pLoop = 1
    out = np.ones_like(N)
    int_i = (a < N) & (N < n_max)
    out[int_i] = np.interp(N[int_i], ncd.sarah_looping.n, ncd.sarah_looping.pLoop)
    # see doc for source of these
    m = -1.47636997617001
    intercept = -2.981924639134709
    out[N >= n_max] = 10**(m * np.log10(N[N>=n_max]) + intercept)
    return out

def load_WLC_looping(ns=None):
    """Compute or load existing values for the bare WLC looping probability,
    in units of base pairs. Saved in files that refer to non-dimensionalized
    units N=L/2*lp (WARNING: not calculated in non-dimensionalized units, so
    they cannot be used unless put back into their own units).

    The default N = [0.005n]*50, [0.02n]*100, [0.2n]*50, [10n]*100.
    There is some overlap, but for values of n where the function is
    numerically stable, they match at the overlap points nicely.

    TODO: fix issue where small n are numerically unstable."""
    default_gprops_dir = Path('csvs/gprops/straight')
    gprops_re = re.compile('gprops_bare_([0-9]+\.?[0-9]*)n_([0-9]+)steps.pkl')
    gprops = []
    all_ns = []
    if ns is not None:
        # operates in "bases" and assumes an lp of 50nm, uses "linker" lengths
        # instead of positions along polymer, so do appropriate conversions
        gprops = [bareWLC_gprop(np.diff(ns)*100/ncg.dna_params['lpb'], unwrap=0)]
        all_ns = [ns]
    else:
        for file in default_gprops_dir.glob('*.pkl'):
            match = gprops_re.match(file.name)
            if match is None:
                continue
            dn, nsteps = match.groups()
            dn = float(dn)
            nsteps = int(nsteps)
            all_ns.append(np.cumsum(np.tile(dn, nsteps)))
            gprops.append(pickle.load(open(file, 'rb')))
    Ploops = []
    for i, gprop in enumerate(gprops):
        dns = np.insert(np.diff(all_ns[i]), 0, all_ns[i][0])
        Ploops.append(BRN_fourier_integrand_splines(dns, unwrap=0,
                rvals=np.array([0]), Bprop=gprop).ravel())
    n, Ploop = np.concatenate(all_ns), np.concatenate(Ploops)
    i = np.argsort(n)
    return 2*default_lp*n[i], Ploop[i]

def fit_persistance_length_to_gaussian_looping_prob(integral, links, unwrap, Nvals=None, Nmin=40):
    """Fit effective persistance length to log-log looping probability vs. chain length (Rmax).
    Nmin is the minimum number of nucleosomes to begin powerlaw fitting to Gaussian chain.
    """
    if Nvals is None:
        Nvals = np.arange(1, len(links)+1)

    ploops = integral[0, :]
    Rmax = convert.Rmax_from_links_unwraps(links, unwraps=unwrap)

    #extract out chain lengths that correspond to Nvals >= Nmin
    inds = Nvals - 1
    inds = inds[Nvals >= Nmin]

    #Gaussian chain limit in log-log space
    ploop_gaussian = np.log(ploops[Nvals >= Nmin])
    Rmax_gaussian = np.log(Rmax[inds])
    m, intercept, rvalue, pvalue, stderr = stats.linregress(Rmax_gaussian, ploop_gaussian)
    print(f'Power law: N^{m}')

    #For Guassian chain, the intercept = (3/2)log(3/(4pi*lp)) -- see Deepti's notes
    lp = 3 / (4*np.pi*np.exp(intercept/np.abs(m)))
    lp = lp * ncg.dna_params['lpb']
    return m, lp


def tabulate_bareWLC_propagators(Ks):
    m = 0
    props = []
    for i, K in enumerate(Ks):
        props.append(propagator.propagator(i, K, m))
    return props

def bareWLC_gprop(links, unwrap, Nvals=None, props=None, **kwargs):
    """Calculate G(K;N) for a bare WLC with the same Rmax as a kinked WLC with
    given linker lengths and unwrapping amount at chain lengths dictated by
    Nvals (# of nucleosomes).

    Parameters
    ----------
    kvals : (29999,) array-like
        k values for which to compute G(k; N) in :math:`bp^{-1}`
    links : (L,) array-like
        bare linker lengths in bp (not including unwrapping), one per nuc in chain. Assumes length
        of chain in len(links).
    unwrap : float
        unwrapping amount in bp. Assumes fixed unwrapping.
    Nvals : array-like
        number of linkers down the chain for which you want the propagator. Defaults to one propagator
        per monomer of the chain.
    props : (29999,) array-like
        list of objects of class 'MultiPoint.propagator', one for each k value.

    Returns
    -------
    gprops : (kvals.size, Nvals.size), array-like
        :math:`G_{00}^{0}(k;N)` for all N values

    """
    num_linkers = len(links)
    #here, Nvals refers to number of nucleosomes; need to convert to units of Rmax / 2lp
    if Nvals is None:
        Nvals = np.arange(1, num_linkers+1)

    inds = Nvals - 1
    #in case we want to create WLCs with different persistance lengths
    lp = kwargs['lp'] if 'lp' in kwargs else default_lp
    Rmax = convert.Rmax_from_links_unwraps(links, unwraps=unwrap) #max WLC chain length in bp
    #select out chain lengths for which propgator needs to be calculated
    Rmax = Rmax[inds]
    #here, Ns refere to number of WLC kuhn lengths
    Ns = Rmax / (2*lp)
    #convert to big K units in Quinn's code
    Ks = (2*lp) * kvals

    #Calculate bare WLC propagators, one for each value of K
    if props is None:
        props_file = Path('csvs/quinn_props_default_kvals.pkl')
        if props_file.exists():
            props = pickle.load(open(props_file, 'rb'))
        else:
            props = tabulate_bareWLC_propagators(Ks)
            pickle.dump(props, open(props_file, 'wb'))

    #only care about G000 for real space green's function
    l0 = 0
    l = 0
    #same shape as my Bprops for comparison
    gprops = np.zeros((Ks.size, Ns.size)).astype('complex')
    for i in range(len(Ks)):
        #only calculate l=0, m=0, j=0 case
        #by default, nlam=10 (this is analogous to l0max, but we only want l0=0 anyway)
        for j, N in enumerate(Ns):
            gprops[i, j] = props[i].get_G(N, l0, l)

    return gprops

def prob_R_given_L(integral, rvals, links, unwrap, Nvals=None):
    """Calculate probability that polymer is of length R (as opposed to R vector).
    Integrate :math:`P(R;L) = 4\pi dR R^2 G(R;L)` from R=0 to R=Rmax and
    check that the answer is 1 for all chain lengths.

    Assumes integral is (rvals.size, Rmax.size)."""

    if Nvals is None:
        Nvals = np.arange(1, len(links)+1)

    inds = Nvals - 1
    #R = r * Rmax
    Rmax = convert.Rmax_from_links_unwraps(links, unwraps=unwrap)
    PRN = np.zeros_like(Nvals).astype('float')

    for i in inds:
        Rvals = rvals * Rmax[i]
        y = (4*np.pi) * Rvals**2 * integral[:, i]
        PRN[i] = splint(0.0, Rmax[i], splrep(Rvals, y))
        print(f'P for {Nvals[i]} nucleosomes = {PRN[i]}')

    return PRN #should be 1 for all N values

def prob_R_in_radius_a_given_L(a, integral, rvals, links, unwrap, Nvals=None):
    """Calculate probability that 2 ends of of polymer fall within some contact radius a.
    Integrate :math:`P(R;L) = 4\pi dR R^2 G(R;L)` from R=0 to R=a, where a is in bp.
    Note: In Sarah's paper, a = L / (2lp) = 0.1, so multiply by 0.1 * 2*lp = 30 bp
    Assumes integral is (rvals.size, Rmax.size)."""

    if Nvals is None:
        Nvals = np.arange(1, len(links)+1)

    inds = Nvals - 1
    Rmax = convert.Rmax_from_links_unwraps(links, unwraps=unwrap)
    PRN = np.zeros_like(Nvals).astype('float')

    for i in inds:
        Rvals = rvals * Rmax[i]
        y = (4*np.pi) * Rvals**2 * integral[:, i]
        PRN[i] = splint(0.0, a, splrep(Rvals, y))

    #1 probability per chain length
    return PRN

def BRN_fourier_integrand(k, R, N, link, unwrap):
    """Fourier integrand for kinked WLC green's function to be passed to quad."""
    B0 = Bprop_k_given_L_fixed_linkers_fixed_unwrap(k, N, link, unwrap)
    return k**2 * special.spherical_jn(0, k*R) * B0.real

def greens_R_given_L_fixed_linkers_fixed_unwraps(r, links, unwrap, kmin=0, kmax=300):
    """Return the unnormalized, real-space Green's function G(R; L) using scipy's
    quad integration. NOTE: This is too slow, so don't ever run this code."""

    num_linkers = len(links)
    if(np.array_equal(np.tile(links[0], num_linkers), links) is False):
        raise ValueError('linker lengths must all be constant')

    #total length of chain in genomic distance; len(Ldna) = number of linkers/nucleosomes
    Ldna = ncg.genomic_length_from_links_unwraps(links, unwraps=unwrap)
    integral = np.zeros_like(Ldna).astype('float')
    #errs = np.zeros_like(integral)

    #for i, r in enumerate(rvals):
    for j, L in enumerate(Ldna):
        N = j+1 #number of nucleosomes, aka Nth propagator
        R = r*L
        sol, err = quad(BRN_fourier_integrand, kmin, kmax, args=(R, N, links[0], unwrap))
        print(f'Computed integral for r={r}, L={L}')
        integral[j] = sol
        #errs[j] = err

    return integral

###}}}

###{{{
# TESTING FUNCTIONS

def test_lmax_convergence(K, N, l, l0, j, l0max=20, tol=10**-8, **kwargs):
    """Determine the maximum l value needed for G(K;N) to converge."""

    # let lmax range from 50 to 500
    lmax = l0max + 30
    ans = get_G(K, N, l, l0, j, l0max, lmax, **kwargs)
    prev = ans - 10
    while(np.abs(ans - prev) > tol):
        # keep increasing lmax by 10 until answers are within specified tolerance
        lmax += 10
        prev = ans
        ans = get_G(K, N, l, l0, j, l0max, lmax, **kwargs)

    print(f'G(K;N) for l={l}, l0={l0}, j={j} converged within tol={tol} at lmax={lmax}')
    return lmax, ans


def save_greens_fixed_links_fixed_unwraps(links, unwrap=0, num_nucs=50, Kprops_bareWLC=None):
    """To process many chains with fixed linker lengths, load in the saved Bprops, do the Fourier
    inversion, and save the output so we can quickly look at them later."""
    Klin = np.linspace(0, 10**5, 20000)
    Klog = np.logspace(-3, 5, 10000)
    Kvals = np.unique(np.concatenate((Klin, Klog)))
    #convert to little k -- units of inverse bp (this results in kmax = 332)
    kvals = Kvals / (2*wlc.default_lp)
    if Kprops_bareWLC is None:
        Kprops_bareWLC = wlc.tabulate_bareWLC_propagators(Kvals)
    print('Tabulated K propagators for bare WLC!')

    for link in links:
        linkers = np.tile(link, num_nucs)
        #for each chain, do fourier inversion integral and save output as .npy file in Bprops directory
        Bprop = np.load(f'csvs/Bprops/{unwrap}unwraps/{link}link/B0_k_given_N_{link}bplinkers_{unwrap}unwraps_50nucs_30000Ks.npy')
        rvals = np.linspace(0.0, 1.0, 1000)
        qprop = wlc.bareWLC_gprop(kvals, linkers, unwrap, props=Kprops_bareWLC) #default: 1000 rvals
        integral = wlc.BRN_fourier_integrand_splines(kvals, linkers, unwrap, Bprop=Bprop, rvals=rvals) #default: 1000 rvals
        qintegral = wlc.BRN_fourier_integrand_splines(kvals, linkers, unwrap, Bprop=qprop, rvals=rvals) #default: 1000 rvals
        #integral takes ~10 min to run, so prob worth saving
        np.save(f'csvs/Bprops/{unwrap}unwraps/{link}link/kinkedWLC_greens_{link}link_{unwrap}unwraps_{len(rvals)}rvals_50nucs.npy', integral, allow_pickle=False)
        np.save(f'csvs/Bprops/{unwrap}unwraps/{link}link/bareWLC_greens_{link}link_{unwrap}unwraps_{len(rvals)}rvals_50nucs.npy', qintegral, allow_pickle=False)
        print(f'Saved G(R;L) for {link}link, {unwrap} unwrap!')

def save_greens_hetero_links_plus_one(links, unwrap=0, Kprops_bareWLC=None):
    """Load in saved Bprop for a heterogenous chain with 2 random linker lengths, link and link+1,
    and calculate/save Green's function."""
    if Kprops_bareWLC is None:
        Kprops_bareWLC = wlc.tabulate_bareWLC_propagators(Kvals)
    print('Tabulated K propagators for bare WLC!')
    link = min(links)
    #for each chain, do fourier inversion integral and save output as .npy file in Bprops directory
    Bprop = np.load(f'csvs/Bprops/{unwrap}unwraps/heterogenous/B0_k_given_N_links{link}or{link+1}_50nucs_30000Ks.npy')
    rvals = np.linspace(0.0, 1.0, 1000)
    qprop = wlc.bareWLC_gprop(kvals, links, unwrap, props=Kprops_bareWLC) #default: 1000 rvals
    integral = wlc.BRN_fourier_integrand_splines(kvals, links, unwrap, Bprop=Bprop, rvals=rvals) #default: 1000 rvals
    qintegral = wlc.BRN_fourier_integrand_splines(kvals, links, unwrap, Bprop=qprop, rvals=rvals) #default: 1000 rvals
    #integral takes ~10 min to run, so prob worth saving
    np.save(f'csvs/Bprops/{unwrap}unwraps/heterogenous/kinkedWLC_greens_links{link}or{link+1}_{len(rvals)}rvals_50nucs.npy', integral, allow_pickle=False)
    np.save(f'csvs/Bprops/{unwrap}unwraps/heterogenous/bareWLC_greens_links{link}or{link+1}_{len(rvals)}rvals_50nucs.npy', qintegral, allow_pickle=False)
    print(f'Saved G(R;L) for {link} or {link+1} link, {unwrap} unwrap!')

# def compare_Quinn_Deepti_props(Ks, Ns, l, l0, j):
#     props = []
#     for i,K in enumerate(Ks):
#         props.append(propagator.propagator(i, K, 1))
#         for N in Ns:
#             lmax, my_ans = wlc.test_lmax_convergence(K, N, l, l0, j)
#             quinn_ans = props[i].get_G(N, l0, l)
#             print(f'G({K};{N}): Quinn -- {quinn_ans}, Deepti -- {my_ans}')
#     return props

###}}}

# from multiprocessing import Pool
# %time
# if __name__ == '__main__':
#     #this is the same-ish range that Quinn used
    # Klin = np.linspace(0, 10**5, 20000)
    # Klog = np.logspace(-3, 5, 10000)
    # Kvals = np.unique(np.concatenate((Klin, Klog)))
    # #convert to little k -- units of inverse bp (this results in kmax = 332)
    # kvals = Kvals / (2*wlc.default_lp)
#     with Pool(31) as pool:
#         #returns a list of 30,000 441 by 441 matrices; don't need to pickle
#         #since function already saves them in csv's
#         gprops = pool.map(partial(wlc.gprop_k_given_link, link=26, unwrap=0), kvals)
#     with Pool(31) as pool:
#         gprops = pool.map(partial(wlc.gprop_k_given_link, link=30, unwrap=0), kvals)
#     with Pool(31) as pool:
#         gprops = pool.map(partial(wlc.gprop_k_given_link, link=245, unwrap=0), kvals)

# if __name__ == '__main__':
#     #this is the same-ish range that Quinn used
#     Kmax = 10**5
#     kmax = Kmax / (2*wlc.default_lp)
#     #suppose chain consists of 50, 36bp linkers -- this is about the length at which the R^2 plot levels out
#     links = np.tile(36, 50)
#     rvals = np.linspace(0.0, 1.0, 100) # R/L
#     with Pool(32) as pool:
#         #returns a list of np arrays, each array is B000(k;N) for each of the Nvals
#         integral = pool.map(partial(wlc.greens_R_given_L_fixed_linkers_fixed_unwraps, links=links, unwrap=0, kmax=kmax), rvals)

#     #integral should now be a matrix where rows are r values and columns are N (or L) values
#     integral = np.array(integral)
#     pickle.dump(integral, open(f'G_R_given_L_36bplinkers_0unwraps_50nucs.p', 'wb'))

#GROW CHAIN FROM 400 TO 500 nucs
# %%time
# from multiprocessing import Pool
# if __name__ == '__main__':
#     #this is the same-ish range that Quinn used
#     Klin = np.linspace(0, 10**5, 20000)
#     Klog = np.logspace(-3, 5, 10000)
#     Kvals = np.unique(np.concatenate((Klin, Klog)))
#     #convert to little k -- units of inverse bp (this results in kmax = 332)
#     kvals = Kvals / (2*wlc.default_lp)
#     Nvals = np.arange(400, 501, 2)
#     #suppose chain consists of 50, 36bp linkers -- this is about the length at which the R^2 plot levels out
#     links = np.tile(50, 500)
#     with Pool(31) as pool:
#         #returns a list of np arrays, each array is B000(k;N) for each of the Nvals
#         bprops50_400to500nucs = pool.map(partial(wlc.Bprop_k_fixed_linkers_fixed_unwrap, links=links, unwrap=0, Nvals=Nvals), kvals)

#     #bprops should now be a matrix where rows are k values and columns are N values
#     bprops50_400to500nucs = np.array(bprops50_400to500nucs)
#     np.save(f'csvs/Bprops/0unwraps/50link/B0_k_given_N_50bplinkers_0unwraps_400to500nucs_30000Ks.npy', bprops50_400to500nucs, allow_pickle=False)
#     links = np.tile(42, 500)
#     with Pool(31) as pool:
#         #returns a list of np arrays, each array is B000(k;N) for each of the Nvals
#         bprops42_400to500nucs = pool.map(partial(wlc.Bprop_k_fixed_linkers_fixed_unwrap, links=links, unwrap=0, Nvals=Nvals), kvals)
#     #bprops should now be a matrix where rows are k values and columns are N values
#     bprops42_400to500nucs = np.array(bprops42_400to500nucs)
#     np.save(f'csvs/Bprops/0unwraps/42link/B0_k_given_N_42bplinkers_0unwraps_400to500nucs_30000Ks.npy', bprops42_400to500nucs, allow_pickle=False)
#     links = np.tile(36, 500)
#     with Pool(31) as pool:
#         #returns a list of np arrays, each array is B000(k;N) for each of the Nvals
#         bprops36_400to500nucs = pool.map(partial(wlc.Bprop_k_fixed_linkers_fixed_unwrap, links=links, unwrap=0, Nvals=Nvals), kvals)
#     #bprops should now be a matrix where rows are k values and columns are N values
#     bprops36_400to500nucs = np.array(bprops36_400to500nucs)
#     np.save(f'csvs/Bprops/0unwraps/36link/B0_k_given_N_36bplinkers_0unwraps_400to500nucs_30000Ks.npy', bprops36_400to500nucs, allow_pickle=False)


#HOMOGENOUS CHAINS
# %%time
# from multiprocessing import Pool
# if __name__ == '__main__':
#     #this is the same-ish range that Quinn used
#     Klin = np.linspace(0, 10**5, 20000)
#     Klog = np.logspace(-3, 5, 10000)
#     Kvals = np.unique(np.concatenate((Klin, Klog)))
#     #convert to little k -- units of inverse bp (this results in kmax = 332)
#     kvals = Kvals / (2*wlc.default_lp)
#     #suppose chain consists of 50, 36bp linkers -- this is about the length at which the R^2 plot levels out
#     links = np.tile(38, 50)
#     with Pool(31) as pool:
#         #returns a list of np arrays, each array is B000(k;N) for each of the Nvals
#         bprops38 = pool.map(partial(wlc.Bprop_k_fixed_linkers_fixed_unwrap, links=links, unwrap=20), kvals)
#     #bprops should now be a matrix where rows are k values and columns are N values
#     bprops38 = np.array(bprops38)
#     np.save(f'csvs/Bprops/20unwraps/38link/B0_k_given_N_38bplinkers_20unwraps_50nucs_30000Ks.npy', bprops38, allow_pickle=False)

# %%time
# from multiprocessing import Pool
# if __name__ == '__main__':
#     #this is the same-ish range that Quinn used
#     Klin = np.linspace(0, 10**5, 20000)
#     Klog = np.logspace(-3, 5, 10000)
#     Kvals = np.unique(np.concatenate((Klin, Klog)))
#     #convert to little k -- units of inverse bp (this results in kmax = 332)
#     kvals = Kvals / (2*wlc.default_lp)
#     rvals = np.linspace(0.0, 1.0, 1000)
#     Kprops_bareWLC = wlc.tabulate_bareWLC_propagators(Kvals)
#     print('Tabulated K propagators for bare WLC!')
#     unwrap = 0
#     ### Sample heterogenous linkers from an entire period
#     links36to47 = np.random.randint(36, 48, 50)
#     with Pool(31) as pool:
#         #returns a list of B000(k;N=1) for each of the kvals
#         bprops36to47 = pool.map(partial(wlc.Bprop_k_given_L, links=links36to47, unwraps=0), kvals)
#     #bprops should now be a matrix where rows are k values and columns are N values (50 of them)
#     bprops36to47 = np.array(bprops36to47)
#     np.save(f'csvs/Bprops/0unwraps/heterogenous/B0_k_given_N_links36to47_50nucs_30000Ks.npy', bprops36to47, allow_pickle=False)
#     np.save(f'csvs/Bprops/0unwraps/heterogenous/linker_lengths_36to47_50nucs.npy', links36to47, allow_pickle=False)
#     qprop36to47 = wlc.bareWLC_gprop(kvals, links36to47, unwrap, props=Kprops_bareWLC)
#     integral36to47 = wlc.BRN_fourier_integrand_splines(kvals, links36to47, unwrap, Bprop=bprops36to47, rvals=rvals) #default: 1000 rvals
#     qintegral36to47 = wlc.BRN_fourier_integrand_splines(kvals, links36to47, unwrap, Bprop=qprop36to47, rvals=rvals) #default: 1000 rvals
#     #integral takes ~10 min to run, so prob worth saving
#     np.save(f'csvs/Bprops/{unwrap}unwraps/heterogenous/kinkedWLC_greens_links36to47_{len(rvals)}rvals_50nucs.npy', integral36to47, allow_pickle=False)
#     np.save(f'csvs/Bprops/{unwrap}unwraps/heterogenous/bareWLC_greens_links36to47_{len(rvals)}rvals_50nucs.npy', qintegral36to47, allow_pickle=False)
#     print(f'Saved G(R;L) for 36to47 links, {unwrap} unwrap!')

#     #and do it again!
#     links36to47_r2 = np.random.randint(36, 48, 50)
#     with Pool(31) as pool:
#         #returns a list of B000(k;N=1) for each of the kvals
#         bprops36to47 = pool.map(partial(wlc.Bprop_k_given_L, links=links36to47_r2, unwraps=0), kvals)
#     #bprops should now be a matrix where rows are k values and columns are N values (50 of them)
#     bprops36to47_r2 = np.array(bprops36to47_r2)
#     np.save(f'csvs/Bprops/0unwraps/heterogenous/B0_k_given_N_links36to47_r2_50nucs_30000Ks.npy', bprops36to47_r2, allow_pickle=False)
#     np.save(f'csvs/Bprops/0unwraps/heterogenous/linker_lengths_36to47_r2_50nucs.npy', links36to47_r2, allow_pickle=False)
#     qprop36to47_r2 = wlc.bareWLC_gprop(kvals, links36to47_r2, unwrap, props=Kprops_bareWLC)
#     integral36to47_r2 = wlc.BRN_fourier_integrand_splines(kvals, links36to47_r2, unwrap, Bprop=bprops36to47_r2, rvals=rvals) #default: 1000 rvals
#     qintegral36to47_r2 = wlc.BRN_fourier_integrand_splines(kvals, links36to47_r2, unwrap, Bprop=qprop36to47_r2, rvals=rvals) #default: 1000 rvals
#     #integral takes ~10 min to run, so prob worth saving
#     np.save(f'csvs/Bprops/{unwrap}unwraps/heterogenous/kinkedWLC_greens_links36to47_r2_{len(rvals)}rvals_50nucs.npy', integral36to47_r2, allow_pickle=False)
#     np.save(f'csvs/Bprops/{unwrap}unwraps/heterogenous/bareWLC_greens_links36to47_r2_{len(rvals)}rvals_50nucs.npy', qintegral36to47_r2, allow_pickle=False)
#     print(f'Saved G(R;L) for 36to47 links, {unwrap} unwrap round 2!')
