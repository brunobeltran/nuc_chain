# nuc\_chain

[![Documentation
Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest)

Model chromatin as a chain of nucleosomes.

The nuc\_chain package contains routines to calculate various configurational
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

## Usage

This package can be pip installed

```bash
pip install nuc_chain
```

For detailed usage instructions, simply build the documentation using
sphinx by cloning the repository and using `make` in the `doc` directory.
