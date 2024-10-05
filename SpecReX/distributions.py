'''
Modified Code from Private Code Repository:
Multiple Different Black Box Explanations for Image Classifiers by Chockler et al.
https://arxiv.org/abs/2309.14309

'''

from enum import Enum
import numpy as np
from scipy.stats import binom

Distribution = Enum("Distribution", ["Binomial", "Uniform"])

def str2distribution(d: str) -> Distribution:
    if d == "binom":
        return Distribution.Binomial
    elif d == "uniform":
        return Distribution.Uniform
    else:
        return Distribution.Uniform
   
def random_pos(d: Distribution | None, *args):
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