# -*- coding: utf-8 -*-
r"""Decorator functions for WLC fluctuations module

Implements useful caching functions to save propogators and minimize extra calculations.
"""

import numpy as np
import pickle
from pathlib import Path

def cache_pickle(f):
    name_of_function_f_as_string = f.__name__
    my_file = Path(f'csvs/{name_of_function_f_as_string}.p')
    def f_prime(*args, **kwargs):
        if args not in f_prime.cache:
            f_prime.cache[args] = f(*args, **kwargs)
            pickle.dump(f_prime.cache, open(f'csvs/{name_of_function_f_as_string}.p', 'wb'))
        return f_prime.cache[args]

    if my_file.is_file():
        f_prime.cache = pickle.load(open(f'csvs/{name_of_function_f_as_string}.p', 'rb'))
    else:
        f_prime.cache = {}

    return f_prime

def cache(f):
    def f_prime(*args, **kwargs):
        if args not in f_prime.cache:
            f_prime.cache[args] = f(*args, **kwargs)
        return f_prime.cache[args]
    f_prime.cache = {}
    return f_prime


# @cache
# def f(x):
#     return x + 1
#
# @cache is same as saying
# f = cache(f)

# import utils.py
# utils.cache_args_only exists
#
# @cache_args_only
# def f(x):
#    pass
#
# now f "=" cache_args_only(f) "=" f_prime <- f
# and f_prime.cache is the dict

# def f(x):
#     a = lambda y: y + x
#     return a

# def g(x):
#     f.cache[key]



def random_positions_in_lattice(count, lattice_size):
    """Insert n steric objects (of width 1) into a discrete lattice of
    lattice_size (uniform potential of insertion).

    Equivalent to np.sort(np.random.shuffle(np.arange(lattice_size)))[:n]

    Notes
    -----
    Translated from the following code on stack overflow
    (https://stackoverflow.com/questions/16000196/java-generating-non-repeating-random-numbers)
    from the book "programming pearls" pg 127.
    public static int[] sampleRandomNumbersWithoutRepetition(int start, int end, int count) {
        Random rng = new Random();

        int[] result = new int[count];
        int cur = 0;
        int remaining = end - start;
        for (int i = start; i < end && count > 0; i++) {
            double probability = rng.nextDouble();
            if (probability < ((double) count) / (double) remaining) {
                count--;
                result[cur++] = i;
            }
            remaining--;
        }
        return result;
    }
    """
    result = np.zeros((count,))
    cur = 0
    remaining = lattice_size
    i = 0
    while True:
        if np.random.rand() < float(count)/float(remaining):
            count -= 1
            result[cur] = i
            cur += 1
        remaining -= 1
        if not (i < lattice_size and count > 0):
            break
        i += 1
    return result
