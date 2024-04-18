# TODO write wrapper for discrete distributions

from enum import Enum
import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom, betabinom
from numba import njit

Distribution = Enum("Distribution", ["Binomial", "Uniform", "BetaBinomial", "Adaptive"])


@njit()
def sparse_softmax(pixel_ranking, invert):
    arr = pixel_ranking.flatten()
    if invert:
        arr = np.array([1 - x if x != 0.0 else 0.0 for x in arr])
    top = np.array([np.exp(x) if x != 0.0 else 0.0 for x in arr])
    out = top / np.sum(top)
    out[np.isnan(out)] = 0.0
    return out


def adaptive(pixel_ranking: NDArray, invert):
    flat = sparse_softmax(pixel_ranking, invert)
    sample = np.random.choice(flat.size, p=flat)
    x, y = np.unravel_index(sample, pixel_ranking.shape)
    return x, y

def pos_adaptive(pos_ranking: NDArray, invert):
    flat = sparse_softmax(pos_ranking, invert)
    sample = np.random.choice(flat.size, p=flat)
    return sample


def str2distribution(d: str) -> Distribution:
    if d == "binom":
        return Distribution.Binomial
    elif d == "uniform":
        return Distribution.Uniform
    elif d == "betabinom":
        return Distribution.BetaBinomial
    elif d == "adaptive":
        return Distribution.Adaptive
    else:
        return Distribution.Uniform


def random_coords(d: Distribution | None, *args, pixel_ranking=None, invert=True):
    if d == Distribution.Adaptive:
        if pixel_ranking is None:
            return random_coords(Distribution.Uniform, *args)
        else:
            return adaptive(pixel_ranking, invert)

    start, stop, *dist_args = args[0]
    start += 1
    if stop - start < 2:
        return
    if stop - start == 2:
        return start + 1
    if d == Distribution.Uniform or d is None:
        return np.random.randint(start, stop)

    if d == Distribution.Binomial:
        return binom(stop - start - 1, dist_args).rvs() + start

    if d == Distribution.BetaBinomial:
        stop -= 1
        # not robust
        alpha = dist_args[0][0]
        beta = dist_args[0][1]
        return betabinom(stop - start, alpha, beta).rvs() + start
    
def random_pos(d: Distribution | None, *args, pos_ranking = None, invert = True):
    if d == Distribution.Adaptive:
        if pos_ranking is None:
            return random_pos(Distribution.Uniform, *args)
        else:
            return pos_adaptive(pos_ranking, invert)

    start, stop, *dist_args = args[0]
    start += 1
    if stop - start < 2:
        return
    if stop - start == 2:
        return start + 1
    if d == Distribution.Uniform or d is None:
        return np.random.randint(start, stop)
    
    if d == Distribution.Binomial:
        return binom(stop - start - 1, dist_args).rvs() + start