import inspect
from pathlib import Path

import pandas as pd
import numpy as np
import pickle

data_dir = Path(inspect.getfile(inspect.currentframe())).parent
data_dir = data_dir.resolve()

spokes_manual_deepti_2018_05_31 = pd.read_csv(data_dir / Path('spokes_manual_deepti_2018_05_31.csv'))
r"""pd.DataFrame: some carbons Deepti extracted that can be used to fit DNA's
orientation around the nucleosome core particle.

Notes
-------

The columns of the DataFrame are

"basepair id" : int
    corresponds to 'i' when calling geometry.H
"base id" : int
    whether on the watson (0) or crick (1) strand. strand name (i.e. which is
    watson and which is crick) arbitrarily chosen.
"deepti id" : int
    arbitrary order in which pymol exported the bases, up the watson strand
    then back down the crick strand in the opposite direction.
"x": float
    x-coordinate of the terminal carbon of that base
"y": float
    y-coordinate of the terminal carbon of that base
"z": float
    z-coordinate of the terminal carbon of that base
"""
spokes_manual_deepti_2018_05_31.set_index(['deepti id'], inplace=True)
#retrieve dictionary of 441 by 441 M kink matrices
Mdict_from_unwraps = pickle.load(open(data_dir / Path('Mdict_from_unwraps.p'), 'rb'))

class SarahLooping:
    """Delayed loading for the looping probability of a bare WLC with a capture
    radius of a=0.1b (1/10th Kuhn lengths), as tabulated by Sarah.

    An instance of this singleton class will have field n and pLoop,
    corresponding to the n values along which the looping probability was
    tabulated (the units of N=L/b are "# of Kuhn lengths"), and the looping
    probability itself, on those points."""
    def __init__(self, file=data_dir/Path('./sarah_loop_prob_a_p1.txt')):
        self.file = file
        self.initialized = False

    @property
    def n(self):
        if not self.initialized:
            self.initialize()
        return self.df.n

    @property
    def pLoop(self):
        if not self.initialized:
            self.initialize()
        return self.df.pLoop

    def initialize(self):
        self.df = pd.read_csv(self.file, header=None, sep='\t')
        self.df.columns = ['n', 'pLoop']

sarah_looping = SarahLooping()

