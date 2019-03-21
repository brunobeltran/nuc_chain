import multiprocessing
from multiprocessing import Pool
from pathlib import Path
import itertools

import numpy as np
import pandas as pd

from nuc_chain import fluctuations as wlc
from nuc_chain import geometry as ncg

mus = np.arange(31, 60)
sigmas = [0]
unwraps = [0]

r2_format_string = 'csvs/r2/r2-{fluct}-{mode}-mu_{mu}-sigma_{sigma}-{unwrap}unwraps.csv'
def save_r2s(param, num_chains=None, num_linkers=None, fluct=True, mode='box',
             force_override=False, **kwargs):
    f"""Tabulates r2 for a bunch of chains with fixed parameters.

    Parameters
    ----------
    params : Tuple[int, int, int]
        mu (average linker length), sigma (linker length variability (not
        stddev), and unwrap (number of base pairs unwrapped from nucleosome
    num_chains : int
        number of chains (repeats) to tabulate r2 for.
    num_linkers : int
        how long (in # linkers) each chain should be
    fluct : bool
        whether or not to include fluctuations in r2 calculations. otherwise
        linkers are rigid rods
    mode : string
        options are 'box', for linkers uniformly from mu-sigma/2 to mu+sigma/2,
        and 'exp', for linkers exponentially distributed (so sigma := mu)
    force_override : bool
        whether or not to recompute file if it already exists. this is mostly
        useful if you're increasing num_chains or num_nucleosomes.

    Notes
    -----

    Writes a pandas.DataFrame.to_csv with format string

    >>> {r2_format_string}

    where fluct is 'fluct' or 'geom'.
    """
    mu, sigma, unwrap = param
    fluct = 'fluct' if fluct else 'geom'
    file_name = r2_format_string.format(fluct=fluct, mode=mode, mu=mu, sigma=sigma, unwrap=unwrap)

    if Path(file_name).exists() and not force_override:
        return
    Path(file_name).touch()

    if mode == 'box':
        if not fluct:
            num_chains = num_chains if num_chains else 400
            num_linkers = num_linkers if num_linkers else 7500
            df = ncg.tabulate_r2_heterogenous_chains_by_variance(num_chains, num_linkers, [sigma], mu=mu, unwraps=unwrap, **kwargs)
        elif fluct:
            num_chains = num_chains if num_chains else 50
            num_linkers = num_linkers if num_linkers else 7500
            df = wlc.tabulate_r2_heterogenous_fluctuating_chains_by_variance(num_chains, num_linkers, [sigma], mu=mu, unwraps=unwrap, **kwargs)
    elif mode == 'exp':
        sigma = mu # by definition
        raise NotImplementedError("this is implemented, just need to go dig up what function to call.")
        if not fluct:
            pass
        elif fluct:
            pass
    else:
        raise NotImplementedError("Invalid mode (linker variance type)")

    df['variance'] = sigma
    df.to_csv(file_name)

pool_size = multiprocessing.cpu_count() - 1
with Pool(processes=pool_size) as p:
    p.map(save_r2s, itertools.product(mus, sigmas, unwraps))
