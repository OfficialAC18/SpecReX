import numpy as np
from scipy.stats import binom
import pytest

import warnings
warnings.filterwarnings("ignore")

from ReX.distributions import str2distribution, random_pos

#Start and End point for the function
start = 0
end = 891

#distribution args (Same as SpecReX)
dist_args_binom = [0.7]
dist_args_uni = []


def split_pos(start, stop, dist = 'uniform', dist_args = []):
    start += 1
    if stop - start < 2:
        return
    if stop - start == 2:
        return start + 1
    
    if dist == 'uniform':
        return np.random.randint(start, stop)
    
    if dist == 'binom':
        return binom(stop - start - 1,dist_args).rvs() + start
        

@pytest.mark.parametrize("distribution",[
    "uniform",
    "binom"
])



def test_sample_splits(distribution):

    #Set numpy seed
    np.random.seed(42)

    if distribution == 'uniform':
        #Generate 3 points from the uniform distribution (Split positions)
        point_1 = split_pos(start, end, dist = 'uniform', dist_args = dist_args_uni)
        point_2 = split_pos(start, point_1-1, dist = 'uniform', dist_args = dist_args_uni)
        point_3 = split_pos(point_1+1, end, dist = 'uniform', dist_args = dist_args_uni)

    else:
        #Generate 3 points from the binomial distribution (Split positions)
        point_1 = split_pos(start, end, dist = 'binom', dist_args = dist_args_binom)
        point_2 = split_pos(start, point_1-1, dist = 'binom', dist_args = dist_args_binom)
        point_3 = split_pos(point_1+1, end, dist = 'binom', dist_args=dist_args_binom)

    #Reset seed
    np.random.seed(42)
    d = str2distribution(distribution)

    if distribution == 'uniform':
        dist_args = dist_args_uni
    else:
        dist_args = dist_args_binom
    
    #Generate points
    point_1_fn = random_pos(d,[start, end, dist_args])
    point_2_fn = random_pos(d,[start, point_1_fn-1, dist_args])
    point_3_fn = random_pos(d,[point_1_fn + 1,end, dist_args])

    #Compare the points
    assert point_1 == point_1_fn
    assert point_2 == point_2_fn
    assert point_3 == point_3_fn



