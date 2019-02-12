from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from nuc_chain import fluctuations as wlc
from nuc_chain import geometry as ncg

mus = np.arange(39, 57)
sigmas = [0]
unwraps = [0]

def save_r2s(param):
    mu, sigma, unwrap = param
    file_name = f'csvs/r2/r2-fluctuations-mu_{mu}-sigma_{sigma}_{unwrap}unwraps-random_phi.csv'
    if Path(file_name).exists():
        return
    Path(file_name).touch()
    dff = wlc.tabulate_r2_heterogenous_fluctuating_chains_by_variance(25, 7500, [sigma], mu=mu, unwraps=0, random_phi=True)
    dff.to_csv(file_name)

pool_size=31
with Pool(processes=pool_size) as p:
    p.map(save_r2s, itertools.product(mus, sigmas, unwraps))
# for link in links:
#     save_r2s(mean_linker_length)
