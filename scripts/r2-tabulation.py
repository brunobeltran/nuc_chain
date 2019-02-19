from multiprocessing import Pool
from pathlib import Path
import itertools

import numpy as np
import pandas as pd

from nuc_chain import fluctuations as wlc
from nuc_chain import geometry as ncg

#WARNING: change the file name format string if you change the simulation to
# use exponentially distributed chains instead, or add/remove kwargs like
# random_phi
mus = [41]
sigmas = np.arange(42)
unwraps = [0]

def save_r2s(param, force_override=False):
    mu, sigma, unwrap = param
    # fluct_file_name = f'csvs/r2/r2-fluctuations-homogenous-link-mu_{mu}-sigma_{sigma}_{unwrap}unwraps.csv'
    geom_file_name = f'csvs/r2/r2-geometrical-box-link-mu_{mu}-sigma_{sigma}_{unwrap}unwraps.csv'
    file_name = geom_file_name
    if Path(file_name).exists() and not force_override:
        return
    Path(file_name).touch()
    # for fluctuating "box" case
    # dff = wlc.tabulate_r2_heterogenous_fluctuating_chains_by_variance(25, 7500, [sigma], mu=mu, unwraps=unwrap, random_phi=False)
    dff['variance'] = sigma
    # dff.to_csv(fluct_file_name)
    # for geometrical "box" case
    dfg = ncg.tabulate_r2_heterogenous_chains_by_variance(400, 7500, [sigma], mu=mu, unwraps=unwrap)
    dfg['variance'] = sigma
    dfg.to_csv(geom_file_name)

pool_size=31
with Pool(processes=pool_size) as p:
    p.map(save_r2s, itertools.product(mus, sigmas, unwraps))
# for link in links:
#     save_r2s(mean_linker_length)
