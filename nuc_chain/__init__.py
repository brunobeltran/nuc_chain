"""Model chromatin as a chain of nucleosomes.

The nuc_chain package contains routines to calculate various configurational
and dynamic properties of a model of chromatin that consists of wormlike-chains
(linker DNA) connecting nucleosomes (approximately cylinders or spheres) at
angles determined by the geometry of the entry/exit angles of DNA into the
nucleosome crystal structure.

This package was developed to investigate the contributions of fluctuations in
the linkers, heterogeneity in the linker length, and "breathing" (partial
unwrapping of the DNA from the core particle) of nucleosomes to the physics
governing chromatin fibers.

The eventual goal is to be able to output expected equilibrium structures (that
can also be used for dynamic simulations) given an organism and a range of
genomic coordinates.


Notes
-----
In this model of chromatin, the double-helical DNA is modeled as a
wormlike-chain with twist (tWLC). It is understood to be bound to nucleosomes
at fixed locations (as might be determined via the lattice_nucleosomes.py
module). The configuration of DNA bound to the nucleosome is approximated to
itself be a helix. The number of wraps of DNA around the histone octamer
determines the entry and exit angles of the DNA. The number of wraps is itself
determined by the relative energy benefit of the DNA sequence of interest being
bound at each nucleosome contact point.

A consequence of Chasles's theorem is that if the amount of DNA between
nucleosomes is constant, the higher order structure of the chain of nucleosomes
is itself a helix, regardless of the specific linker length used.

Goals
-----

Incorporate relevant sources of nucleosome heterogeneity.

- unwrapping heterogeneity
- sliding heterogeneity
- initial binding site heterogeneity
- heterogeneity of "extra unwrapping" due to linker histone fixing the exit
angles, or including (or not) linker globule domains

We care a lot about

- local accessibility to transcription factors

Long term

- diffusion vs binding/unbinding histone

"""
__version__ = "0.1.0"

import numpy as np

# some global defaults
bp_in_nuc = 147
bp_in_structure = 146
default_w_in = 63 # 10 unwrapped bases
default_w_out = 63 # 10 unwrapped bases
default_Lw = default_w_in + default_w_out
default_unwrap = bp_in_nuc - (default_w_in + default_w_out) - 1
assert(default_unwrap % 2 == 0)
default_uw_in = default_unwrap/2
default_uw_out = default_unwrap/2

helix_param_order = ['r', 'c', 'T', 'b', 'x0', 'y0', 'z0', 'psi0', 'phi', 'theta']
"""Order of parameters names to be passed to all helix functions"""

helix_centered_params = ['r', 'c', 'T', 'b']
"""Subset of parameters to be passed to centered helix functions"""

helix_length_params = ['r', 'c']
"""Subset of helix parameters that must be scaled if units change"""

helix_params_best_fit_bruno_full_2018_05_31 = {'T': 1.7000000000000002,
        'b': 126, 'c': 44.297700263176672, 'phi': 1.6771849694713676,
        'psi0': 1.408727503155174, 'r': 41.192061481066709,
        'theta': -1.7202108711768689, 'x0': 10.47376213526613,
        'y0': -44.531679077103604, 'z0': -47.455752173714558}
"""first fit, w_in = w_out = 10, (doi:10.1038/38444)"""
helix_params_reported_richmond_davey_2003 = {
        'T': 1.67, 'b': 133.6, 'c': 41.181, 'r': 41.9,
        'phi': 0, 'psi0': 0, 'theta': 0, 'x0': 0, 'y0': 0, 'z0': 0
}
"""Dict[str,float]: Values reported in the paper "The structure of DNA in the
nucleosome core".  Richmond and Davey. Nature (2003), (doi:10.1038/38444). The
only parameter they don't report directly is 'c', which they instead report as
pitch of 25.9A.  Richmond and Davey report that 133.6bp of DNA has 1.67
superhelical turns in their fit to the bound part of the nucleosomal DNA. """

def helix_params_angstrom_to_nm(params):
    """Converts all length parameters of the helix in place.

    The PDB file coordinates are in ångströms, so we convert to nm."""
    for p, val in params.items():
        if p in helix_length_params:
            params[p] = val/10
    return params

def helix_params_to_centered_params(params):
    """Extrapolates the parameters of the partial helix fit to get the
    parameters for the coordinates of a hypothetical nucleosome where all 147
    bp associated with it are totally bound.
    """
    params = {p: params[p] for p in helix_centered_params}
    # the helix should basically continue with the same pitch for this much
    # longer
    scaling_factor = bp_in_nuc/params['b']
    params['T'] *= scaling_factor
    params['c'] *= scaling_factor
    params['b'] = bp_in_nuc
    return params


helix_params_best = helix_params_angstrom_to_nm(helix_params_to_centered_params(
        helix_params_reported_richmond_davey_2003
))
"""parameters to use in all calculations."""

hp = helix_params_best
"""shorthand to save typing when scripting"""

nucleosome_bound_dna_twist_density = 10.17
"""float: Richmond and Davey report a mean value of 34.52 degrees of twist per
base pair, or 10.43 bp/twist.

Luger et al. alternatively report 10.2 for their
146bp sequence.

A third paper (Davey et al., J. Mol. Biol. (2002)) reports
10.23 and 10.15bp/turn for three different sequences of length 146bp.

It is interesting to note that the latter numbers correspond more closely to
the measured periodicity of cellular nucleosome arrays via sequence periodicity
of 10.17bp (Travers and Klug, Phil. Trans. R. Soc. Lond. (1987)). Richmond and
Davey point out that this suggests that cells have evolved to optimize for
histones bound to DNA stretched about 1-2bp per 147bp.

Our spokes-angle-fit is currently not good enough to get a number
out.

In conclusion, we use the in vivo effective number. Since we care about in vivo
nucleosome conformations. And it should be positive, since B-DNA is a
right-handed helix."""

naked_dna_twist_density = 10.5
"""float: Depending on the reference, you can see 10.4, 10.5 or 10.6.

e.g. for 10.4

- Shimada and Yamakawa, Macromolecules 1984

e.g. for 10.5

- Essevaz-Roulet, Bockelmann, and Heslot, PNAS 1997
- Richard et al., Journal of Physics: Condensed Matter 2002

e.g. for 10.6

- Moroz and Nelson, PNAS 1997

Wikipedia claims that the DNA is likely to have a regular Watson-Crick
structure in terms of twist in vivo, so we use the average of the consensus
numbers above."""

naked_dna_length_per_base = 0.332
"""float: (nm)"""

naked_dna_thickness = 2.0
"""float: (nm). Elsewhere on Wiki quoted as 2.37nm."""

dna_params = {'tau_n': 2*np.pi/nucleosome_bound_dna_twist_density,
              'tau_d': 2*np.pi/naked_dna_twist_density,
              'lpb': naked_dna_length_per_base,
              'r_dna': naked_dna_thickness/2}
"""Parameters to be passed around together."""
