"""Some tools for simulating particle occupancy on a lattice where they can
bind/unbind.

Written to investigate nucleosome positioning data."""

from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd

move_history = []

class Lattice(object):
    """A n-bin lattice filled with objects of width w at positions \vec{p}.
    Keeps positions ordered."""
    def __init__(self, n, w, p=None):
        self.n = n
        self.w = w
        self.p = p
        if self.p is None:
            self.p = np.array([])
        self.np = len(self.p)

    def __repr__(self):
        return "Lattice<n={},w={}>: ".format(self.n,self.w) + str(self.p)

    def copy(self):
        return Lattice(self.n, self.w, self.p.copy())

    def add_bead(self, pos):
        """Add as bead at lattice position pos."""
        # assert(1 <= pos <= self.n)
        i = np.searchsorted(self.p, pos)
        self.p = np.insert(self.p, i, pos)
        self.np += 1

    def remove_bead(self, i):
        """Delete the i'th bead."""
        self.p = np.delete(self.p, i)
        self.np -= 1

    def move_bead(self, i, offset):
        """Move the i'th bead by offset. Overshooting drops the bead at the end
        of the lattice.

        Should be equivalent to:
            >>> self.p[i] += offset
            >>> self.p.sort()

        Tests:
            In [27]: l = lg.Lattice(10, 2)
            In [28]: l.p = [6,7,8]
            In [29]: l.move_bead(0, 2); print(l)
            Lattice<n=10,w=2>: [7, 8, 8]
            In [30]: l.move_bead(2, -1); print(l)
            Lattice<n=10,w=2>: [7, 7, 8]
            In [31]: l.move_bead(1, -1); print(l)
            Lattice<n=10,w=2>: [6, 7, 8]
            In [32]: l.move_bead(1, -10); print(l)
            Lattice<n=10,w=2>: [1, 6, 8]
            In [33]: l.move_bead(0, 10); print(l)
            Lattice<n=10,w=2>: [6, 8, 10]
        """
        if offset == 0:
            return
        new_pos = max(1, min(self.n, self.p[i] + offset))
        # get index before which you would insert new_pos if just adding bead
        i_new = np.searchsorted(self.p, new_pos)
        # no movement
        if i <= i_new <= i + 1:
            self.p[i] = new_pos
        # make
        # 1, 2, 3, ..., i-1, i  , i+1, i+2, ..., i+X-2, i+X-1, i+X,...
        # into
        # 1, 2, 3, ..., i-1, i+1, i+2, i+3, ..., i+X-1,     i, i+X,...
        elif i + 1 < i_new:
            X = i_new - i
            self.p[i:i+X-1] = self.p[i+1:i+X]
            # we've slid the elements to the left, so the spot to the left
            # of i+X=i_new opens up
            self.p[i_new-1] = new_pos
        # make
        # 1, 2, 3, ..., i-X-1, i-X, i-X+1, ..., i-1,   i, i+1, ...
        # into
        # 1, 2, 3, ..., i-X-1,   i, i-X,   ..., i-2, i-1, i+1,...
        elif i_new < i:
            X = i - i_new
            self.p[i-X+1:i+1] = self.p[i-X:i]
            # we slide elements to the right, so the spot exact at i-X=i_new
            # opens up
            self.p[i_new] = new_pos

    def move_range(self, left, right, offset):
        """Move the left'th thru right'th beads (inclusize) by offset.
        Overshooting drops the beads at the end of the lattice."""
        if offset == 0:
            return
        self.p[left:right+1] = np.maximum(1, np.minimum(self.n,
                self.p[left:right+1] + offset))
        self.p.sort()
        #TODO generalize move_bead
        # new_pos = np.maximum(1, np.minimum(self.n, self.p[left:right+1] + offset))
        # # get index before which you would insert new_pos if just adding bead
        # left_new, right_new = np.searchsorted(self.p, [new_pos[0], new_pos[-1]])
        # # no movement
        # if i <= i_new <= i + 1:
        #     self.p[i] = new_pos
        # # make
        # # 1, 2, 3, ..., i-1, i  , i+1, i+2, ..., i+X-2, i+X-1, i+X,...
        # # into
        # # 1, 2, 3, ..., i-1, i+1, i+2, i+3, ..., i+X-1,     i, i+X,...
        # elif i + 1 < i_new:
        #     X = i_new - i
        #     self.p[i:i+X-1] = self.p[i+1:i+X]
        #     self.p[i_new] = new_pos
        # # make
        # # 1, 2, 3, ..., i-X-1, i-X, i-X+1, ..., i-1,   i, i+1, ...
        # # into
        # # 1, 2, 3, ..., i-X-1,   i, i-X,   ..., i-2, i-1, i+1,...
        # elif i_new < i:
        #     X = i - i_new
        #     self.p[i-X+1:i+1] = self.p[i-X:i]
        #     self.p[i_new] = new_pos

#############{{{
# moves

class AdaptableMove(metaclass=ABCMeta):
    """Shared structure between adaptable monte carlo moves.

    Implement __call__ to subclass.

    Default values in __init__ are
    ```python
        self.adapt_min = 0
        self.adapt_max = init_lattice.np
        self.adapt_step = 1
        self.param = 1
    ```

    By default, easier_moves() === decrement_param(),
                harder_moves() === increment_param().

    Say your move type needs n random variables per move (n can be zero). Then
    override the list random_generators in class scope to contain functions of
    type size -> random_pool (e.g. lambda size: np.random.random_sample(size)).
    Sequential calls to random_source will then cycle through these random
    numbers.

    By default
    >>>random_generators = [np.random.random_sample]
    which generates a single uniform random source for the move type, one value
    for each move (#TODO up to a fixed max pool size).
    """

    random_generators = {'i': np.random.random_sample}

    def __init__(self, init_lattice, n_moves):
        self.init_lattice_ = init_lattice
        self.n_moves = n_moves
        self.adapt_min = 0
        self.adapt_max = init_lattice.np
        self.adapt_step = 1
        self.param = 1
        self.random_pools_ = {name: rv((n_moves,)) for name, rv in
                              self.__class__.random_generators.items()}
        self.rand_i_ = {name: 0 for name in self.random_pools_}

    def decrement_param(self):
        """Decrement self.param on the grid defined by self.adapt_[min/step/max]."""
        if self.adapt_min < self.param:
            self.param -= self.adapt_step

    def increment_param(self):
        """Increment self.param on the grid defined by self.adapt_[min/step/max]."""
        if self.param < self.adapt_max:
            self.param += self.adapt_step

    def easier_moves(self):
        self.decrement_param()

    def harder_moves(self):
        self.increment_param()

    def random_source(self, name=None):
        if name is None:
            name = list(self.random_pools_.keys())[0]
        if name not in self.random_pools_:
            raise KeyError("AdaptableMove requested random value from non-existent pool")
        rv = self.random_pools_[name][self.rand_i_[name]]
        self.rand_i_[name] += 1
        return rv

    @abstractmethod
    def __call__(self, lattice):
        """Returns a modified version of lattice, with the AdaptableMove
        applied. Does not work in place."""
        return lattice.copy()

class SingleSlideMove(AdaptableMove):
    random_generators = {'choice': lambda size: np.random.random_sample(size),
                         'magnitude': lambda size: (2*np.random.randint(0, 2, size=size) - 1)
                                     *np.random.exponential(size=size)}

    # # override to take reasonably large steps by default
    # def __init__(self, lattice, n_moves):
    #     super().__init__(lattice, n_moves)
    #     self.param = np.round(lattice.np/10).astype(int)

    def __call__(self, lattice):
        new_lattice = lattice.copy()
        if lattice.np == 0:
            move_history.append(['slide', (np.nan, np.nan)])
            return new_lattice
        # first a uniform
        choice = np.floor(lattice.np*self.random_source('choice')).astype(int)
        # then an exponential
        offset = np.round(self.param*self.random_source('magnitude'))
        new_lattice.move_bead(choice, offset)
        move_history.append(['slide', (choice, offset)])
        return new_lattice

class JointSlideMove(AdaptableMove):
    """Only the size of the slide is a parameter, the beads that slide together
    are chosen by picking two random points inside the interval uniformly."""
    random_generators = {'left': lambda size: np.random.random_sample(size),
                         'right': lambda size: np.random.random_sample(size),
                         'magnitude': lambda size: (2*np.random.randint(0, 2, size=size) - 1)
                                     *np.random.exponential(size=size)}

    # # override to take reasonably large steps by default
    # def __init__(self, lattice, n_moves):
    #     super().__init__(lattice, n_moves)
    #     self.param = np.round(lattice.np/10).astype(int)

    def __call__(self, lattice):
        new_lattice = lattice.copy()
        if lattice.np == 0:
            move_history.append(['shift', (np.nan, np.nan, np.nan)])
            return new_lattice
        # left and right boundaries ``uniformly''
        left = np.floor(lattice.np*self.random_source('left')).astype(int)
        right = np.floor(lattice.np*self.random_source('right')).astype(int)
        if right < left:
            left, right = right, left
        # then an exponential
        offset = np.round(self.param*self.random_source('magnitude'))
        new_lattice.move_range(left, right, offset)
        move_history.append(['shift', (left, right, offset)])
        return new_lattice

class AddMoveMuVT(AdaptableMove):
    def __call__(self, lattice):
        new_lattice = lattice.copy()
        i = np.ceil(lattice.n*self.random_source('i')).astype(int)
        new_lattice.add_bead(i)
        move_history.append(['add', i])
        return new_lattice

class RemoveMoveMuVT(AdaptableMove):
    def __call__(self, lattice):
        new_lattice = lattice.copy()
        if new_lattice.np == 0:
            move_history.append(['remove', np.nan])
            return new_lattice
        i = np.floor(lattice.np*self.random_source('i')).astype(int)
        new_lattice.remove_bead(i)
        move_history.append(['remove', i])
        return new_lattice

# end moves
#############}}}

#############{{{
# energy functions
def constant_lattice_energy(lattice):
    return 1


class PreBuiltStericEnergy(object):
    def __init__(self, n, dEb=1):
        self.energies = -dEb*np.ones((n,))

    def __call__(self, lattice):
        if np.any(np.diff(lattice.p) < lattice.w):
            return np.inf
        else:
            return np.sum(self.energies[(lattice.p-1).astype(int)])

class energy_sterics_only(PreBuiltStericEnergy):
    pass
# def steric_lattice_energy(lattice, dEb=1):
#     """Calculate the energy of the current lattice state if we're in the grand
#     canonical ensemble and the lattice is in the presense of a bath with
#     chemical potential mu."""
#     if np.any(np.diff(lattice.p) < lattice.w):
#         return np.inf
#     else:
#         return -lattice.np*dEb

class energy_with_period(PreBuiltStericEnergy):
    def __init__(self, n, period, dEp=1, dEb=1):
        self.period = period
        self.dEp = dEp
        self.dEb = dEb
        self.bp = np.arange(n)
        self.energies = dEp*np.cos(2*np.pi*self.bp/self.period)/2 - dEb

class energy_from_gc(PreBuiltStericEnergy):
    def __init__(self, gc_frac, dEb=1):
        # more gc -> more energetic benefit
        self.energies = -dEb*np.power(gc_frac, 2)


class energy_from_genes(PreBuiltStericEnergy):
    def __init__(self, n, txStart, txDirection, nfr=150, dEb=1):
        self.forward_genes = txStart[txDirection > 0]
        self.reverse_genes = txStart[txDirection < 0]
        self.energies = -dEb*np.ones((n,))
        for gene in self.forward_genes:
            imin = max(0, gene - nfr)
            self.energies[imin:gene] = np.inf
        for gene in self.reverse_genes:
            imax = min(n, gene + nfr)
            self.energies[gene:imax] = np.inf

class energy_from_gene_and_gc(PreBuiltStericEnergy):
    def __init__(self, gc_frac, txStart, txDirection, nfr=150, dEb=1):
        self.n = len(gc_frac)
        gc = energy_from_gc(gc_frac, dEb=dEb)
        self.energies = gc.energies
        genes = energy_from_genes(self.n, txStart, txDirection, nfr=nfr)
        self.energies[np.isinf(genes.energies)] = np.inf

class energy_from_genes_with_period(PreBuiltStericEnergy):
    def __init__(self, n, period, txStart, txDirection, nfr=150, dEp=1, dEb=1):
        self.energies = energy_from_genes(n, txStart, txDirection, nfr=nfr, dEb=dEb).energies
        self.energies += energy_with_period(n, period, dEp=dEp, dEb=0).energies

# end energy functions
#############}}}

#NOTE one approach for canonical ensemble simulation: fix the number bound, only slide moves
def mcmc_simple(n_moves, lattice, energy_f=None, moves=None, kbT=1):
    if moves is None:
        moves = [single_slide_move, add_remove_move]
    move_choice = np.random.randint(len(moves), size=(n_moves,))
    mc_choice = np.random.rand(size=(n_moves,))
    energy = energy_f(lattice)
    min_energy = energy
    for i in range(n_moves):
        new_lattice = moves[move_choice[i]](lattice)
        new_energy = energy_f(new_lattice)
        if new_energy < energy or np.exp(kbT*(energy - new_energy)) < mc_choice[i]:
            energy = new_energy
            lattice = new_lattice
        if new_energy < min_energy:
            min_lattice = new_lattice
    return min_lattice, lattice

def mcmc_full(n_moves, init_lattice, mu, energy_f=None, move_types=None, kbT=1,
              adaptation_interval=None, target_acceptance=None,
              save_interval=None):
    """Perform MHMC with the given initial object, energy function, mc moves,
    and temperature. Allow the monte carlo moves to adapt their magnitudes
    every adaptation_interval steps, so they optimize for a
    target_acceptance.

    In order to simulate the grand canonical ensemble, use the mu argument to
    set the chemical potential of the bath and correctly rescale the acceptance
    probabilities to maintain detailed balance.

    Since the default move_types require mu to be set to work correctly, mu is
    a required parameter. Set to None for other ensembles.
    """
    if energy_f is None:
        energy_f = steric_lattice_energy
    if move_types is None:
        # best moves we have to work with
        move_types = [AddMoveMuVT, RemoveMoveMuVT, SingleSlideMove, JointSlideMove]
    if adaptation_interval is None:
        adaptation_interval = np.max([100, np.min([n_moves/100, 10000])])
    if target_acceptance is None:
        target_acceptance = 0.1
    # tune the adaptability parameters for the given lattice and number of
    # moves if necessary
    move_choice = np.random.randint(len(move_types), size=(n_moves,))
    moves = [MoveType(init_lattice, n_moves=np.sum(move_choice==k))
             for k, MoveType in enumerate(move_types)]
    # initialize mc state variables
    lattice = init_lattice
    energy = energy_f(init_lattice)
    # and variables to save state history
    save_i = []
    lattices = []
    energies = []
    lattices += [lattice]
    energies += [energy]
    # choose the success of all the moves up front for vectorization
    mc_choice = np.random.random_sample(size=(n_moves,))
    # initialize adaptation state variables
    j_a = 0 # number of moves completed since last adaptation
    j_s = 0 # number of moves completed since last save of state
    n_accepted = np.zeros((len(move_types),)) # track acceptance rate
    n_attempted = np.zeros((len(move_types),))
    for i in range(n_moves):
        if j_a == adaptation_interval:
            for k, move in enumerate(moves):
                if n_accepted[k]/n_attempted[k] > target_acceptance:
                    move.harder_moves()
                else:
                    move.easier_moves()
            print('Acceptance ratios: ' + str(n_accepted/n_attempted) + ', Energy: ' + str(energy) + ' [' + str(i) + '/' + str(n_moves) + ']')
            # reset adaptation counter. a move will in fact be completed before
            # we increment at end of loop, so 0, not -1
            j_a = 0
            n_accepted[:] = 0
            n_attempted[:] = 0
        new_lattice = moves[move_choice[i]](lattice)
        new_energy = energy_f(new_lattice)
        Emu = 0
        if mu is not None:
            # this code is specific to the grand canonical ensemble, assuming
            # we choose insert move or delete move randomly then the location,
            # in order to keep detailed balance, we need (in terms of lattice)
            # insertion => prob = n/(np+1)exp{-kbT*(-mu + new_energy - old)}
            # deletion => prob = np/n*exp{-kbT*(mu + new_energy - old)}
            #NOTE: choosing the lattice site first is very inefficient since
            # there are far fewer particles than lattice sites
            if new_lattice.np > lattice.np: # insertion
                Emu = -np.log(lattice.n/(lattice.np+1))/kbT - mu
            elif new_lattice.np < lattice.np: # deletion
                Emu = -np.log(lattice.np/lattice.n)/kbT + mu
        move_history[-1].append(False)
        if new_energy < energy or mc_choice[i] < np.exp(-kbT*(Emu + new_energy - energy)):
            move_history[-1][-1] = True
            energy = new_energy
            lattice = new_lattice
            if save_interval is None:
                save_i.append(i + 1) # so lattices[0] == init_lattice
                energies += [new_energy]
                lattices += [new_lattice]
            n_accepted[move_choice[i]] += 1
        n_attempted[move_choice[i]] += 1
        if j_s == save_interval:
            save_i.append(i + 1) # so lattices[0] == init_lattice
            energies += [energy]
            lattices += [lattice]
            # reset save_interval counter. no moves will be completed before we
            # increment at end of loop, so to compensate, make -1
            j_s = -1
        j_a += 1
        j_s += 1
    return save_i, energies, lattices

def dyad_occupancy_from_lattices(lattices, noise=0):
    occupancy = np.zeros((lattices[0].n,))
    for lattice in lattices:
        # lattice.p in [1, lattice.n]
        err = np.round(noise*np.random.random_sample(size=lattice.np))
        ix = np.array(lattice.p) - 1 + err
        ix[ix >= lattice.n] = lattice.n - 1
        ix[ix < 0] = 0
        occupancy[ix.astype(int)] += 1
    return occupancy

def acf_from_occupancy(occ, max_lag=None):
    if max_lag is None or max_lag > len(occ):
        max_lag = len(occ)
    occdf = pd.Series(occ)
    return np.array([occdf.autocorr(i) for i in range(max_lag)])

def move_history_frame(moves_df=None):
    if moves_df is None:
        moves_df = move_history
    moves_df = pd.DataFrame(moves_df)
    moves_df.columns = ['move_type', 'parameters', 'success']
    return moves_df

def experiment_uniform_chromosome(N=1000000):
    save_i, energies_big, lattices_big = lg.mcmc_full(10000000,
            init_lattice_big, mu, save_interval=100)
    occupancy = dyad_occupancy_from_lattices(lattices)
    plt.figure()
    plt.plot(occupancy)
    plt.xlabel('genomic coordinate (bp)')
    ocupancy = dyad_occupancy_from_lattices(lattices)
    plt.figure()
    plt.plot(occupancy)
    plt.xlabel('genomic coordinate (bp)')
    plt.ylabel('occupancy, MC-time averaged, 10^6 save points')
    plt.title('Occupancy. big, uniform chromosome, nucleosome MC')
    plt.savefig('plots/big_uniform_occupancy.pdf')
    plt.xlim([0, 10000])
    plt.savefig('plots/big_uniform_occupancy_zoom.pdf')

    acf = acf_from_occupancy(occupancy, max_lag=10000)
    plt.figure()
    plt.plot(acf)
    plt.title('ACF(Occupancy). big, uniform chromosome, nucleosome MC')
    plt.ylabel('ACF(Occupancy)')
    plt.xlabel('lag (bp)')
    plt.savefig('plots/big_uniform_acf.pdf')
    plt.ylabel('occupancy, MC-time averaged, 10^6 save points')
    plt.title('Occupancy. big, uniform chromosome, nucleosome MC')
    plt.savefig('plots/big_uniform_occupancy.pdf')
    plt.xlim([0, 10000])
    plt.savefig('plots/big_uniform_occupancy_zoom.pdf')

    acf = acf_from_occupancy(occupancy, max_lag=10000)
    plt.figure()
    plt.plot(acf)
    plt.title('ACF(Occupancy). big, uniform chromosome, nucleosome MC')
    plt.ylabel('ACF(Occupancy)')
    plt.xlabel('lag (bp)')
    plt.savefig('plots/big_uniform_acf.pdf')





