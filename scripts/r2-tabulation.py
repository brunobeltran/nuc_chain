from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from nuc_chain import fluctuations as wlc
from nuc_chain import geometry as ncg

mu = 46
sigmas = np.arange(0, 46)
unwrap = 0
def save_r2s(sigma):
    file_name = f'csvs/r2/r2-fluctuations-mu_{mu}-sigma_{sigma}_{unwrap}unwraps.csv'
    if Path(file_name).exists():
        return
    Path(file_name).touch()
    dff = wlc.tabulate_r2_heterogenous_fluctuating_chains_by_variance(25, 7500, [sigma], mu=mu, unwraps=0)
    dff.to_csv(file_name)

pool_size=31
with Pool(processes=pool_size) as p:
    p.map(save_r2s, sigmas)
# for link in links:
#     save_r2s(mean_linker_length)
