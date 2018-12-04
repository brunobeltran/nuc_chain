"""Math utilities and other miscellaneous calculations."""
import numpy as np

def r2(r, axis=None):
    """Calculate the instantaneous r^2 along a given axis.

    r.shape == (N,3) ==> r2(r, axis=0) would give the r^2"""
    zero = [slice(None)]*len(r.shape) # [:,:,...,:]
    zero[axis] = slice(0, 1) # [:,:,..,0,..,:]
    return np.sum(np.power(r - r[zero], 2), axis=1-axis)

def links_rmax(links):
    """Converts the list of linker lengths in the chain to a list of
    coordinates in units of distance-along-the-chain, one for each nucleosome.
    Reverse-diff.

    Parameters
    ----------
    links : (N,) array_like
        List of linkers in units of distance-along-the-chain.

    Returns
    -------
    rmax : (N+1,) array_like
        List of distances along the chain of each nucleosome.
    """
    rmax = np.zeros((len(links)+1,))
    rmax[1:] = np.cumsum(links)
    return rmax

def rolling(df):
    return df.rolling(window_size).mean()

