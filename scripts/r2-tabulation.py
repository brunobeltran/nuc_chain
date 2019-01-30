from multiprocessing import Pool

import numpy as np
import pandas as pd

from nuc_chain import fluctuations as wlc
from nuc_chain import geometry as ncg

#calculate geometrical and fluctuating r2 for entire period [31,51]bp inclusive

#already calculate r2 for 36, 38, 42bp with and without 20bp unwrapping
# links = np.concatenate((np.arange(31, 36), [37, 39, 40, 41], np.arange(43, 52)))

sigmas = np.arange(0, 11)
def save_r2s(sigma):
    dfg = ncg.tabulate_r2_heterogenous_chains_by_variance(100, 7500, [sigma], mu=41, random_phi=2*np.pi)
    dfg.to_csv(f'csvs/r2/r2-geometrical-mu_41-sigma_{sigma}_0unwraps_random-phi.csv')
    # dff = wlc.tabulate_r2_heterogenous_fluctuating_chains_homogenous(25, 7500, mu=link, unwraps=0)
    # dff.to_csv(f'csvs/r2/r2-fluctuations-homogenous-link-mu_{link}-0unwraps.csv')
    # kuhnsf = wlc.calculate_kuhn_length_from_fluctuating_r2(dff, link, 7500, unwraps=0)
    # np.save(f'csvs/r2/kuhns-fluctuations-mu_{link}-sigma_0_10_0unwraps-2.npy', kuhnsf)
    # print(f'Calculated r2 + kuhn length for mu={link}')

pool_size=31
with Pool(processes=pool_size) as p:
    p.map(save_r2s, sigmas)
# for link in links:
#     save_r2s(mean_linker_length)
