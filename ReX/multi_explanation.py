#!/usr/bin/env python

"""generate multiple explanations from a responsibility landscape <pixel_ranking>"""

from itertools import combinations
import numpy as np
from numpy.typing import NDArray
from numpy.random import randn
from numba import njit
from tqdm import trange

from ReX.model_funcs import Shape
from ReX.logger import logger
from ReX.ranking import linear_search, neighbours, spatial_search


@njit
def dice(im1, im2):
    """calculates dice coefficient between two numpy arrays of the same dimensions"""
    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return 0
    intersection = np.logical_and(im1, im2)
    return 2.0 * intersection.sum() / im_sum


def extract(explanations):
    """extract multiple explanations from a list of explanations"""
    out = []
    size = len(explanations)
    areas = np.zeros(size)

    combs = list(combinations(np.arange(0, size, 1), 2))
    for comb in combs:
        im1 = explanations[comb[0]]
        im2 = explanations[comb[1]]
        out.append((comb, dice(im1, im2)))
        a1 = np.count_nonzero(im1)
        a2 = np.count_nonzero(im2)
        if areas[comb[0]] == 0.0:
            areas[comb[0]] = a1
        if areas[comb[1]] == 0.0:
            areas[comb[1]] = a2

    mat = np.zeros(size * size).reshape((size, size))
    for x, y in out:
        if y > 0.0:
            if areas[x[0]] < areas[x[1]]:
                mat[x[0], x[1]] += 1
            else:
                mat[x[1], x[0]] += 1

    results = mat.sum(axis=0)

    for v in np.argsort(results)[::-1]:
        if np.count_nonzero(mat) == 0:
            break
        mat[v, :] = 0
        mat[:, v] = 0
        r = mat.sum(axis=0)
        results -= r

    final = np.where(results == 0)
    return final, len(final[0]), areas


def overlap(exp1, exp2):
    """check overlap between two numpy arrays"""
    gt = len(np.where(exp1 + exp2 > 0.0)[0])
    ov = len(np.where(exp1 + exp2 == 2.0)[0])

    return ov / gt


def random_steps(r, c, step_size, lim_r, lim_c):
    """takes random steps within a landscape"""
    new_r = int(r + randn(1) * step_size)
    while new_r < 0 or new_r > lim_r:
        new_r = int(r + randn(1) * step_size)
    new_c = int(c + randn(1) * step_size)
    while new_c < 0 or new_c > lim_c:
        new_c = int(c + randn(1) * step_size)

    return new_r, new_c


def multi_spotlight(
    img_array,
    prediction_func,
    targets,
    pixel_ranking,
    mask_value,
    initial_size,
    size_eta,
    step_size,
    spotlights,
    chunk_size,
):
    """launches multiple spotlight searches over a responsibility landscape"""
    results = []
    logger.info("find global maximum first")
    exp1 = linear_search(img_array, prediction_func, targets, pixel_ranking, mask_value, chunk_size)
    results.append(exp1)
    logger.info("starting spotlight search")
    for _ in trange(spotlights - 1):
        results.append(
            spotlight_search(
                img_array,
                prediction_func,
                targets,
                pixel_ranking,
                mask_value,
                initial_size,
                size_eta,
                step_size,
                np.mean,
                chunk_size,
            )
        )
    results = list(filter(lambda mat: mat is not None and np.sum(mat) > 0, results))
    return results


def spotlight_search(
    img_array: NDArray,
    prediction_func,
    targets: NDArray,
    pixel_ranking: NDArray[np.float32],
    mask_value: int,
    radius: int,
    radius_eta: float,
    step_size: int,
    obj_func,
    chunk_size,
    r=None,
    c=None,
    total_steps_remaining=20,
):
    """performs a spotlight search over a responsibility landscape"""
    shape = Shape(img_array.shape)

    if r is None:
        r = np.random.randint(radius // 2, shape.width)
    if c is None:
        c = np.random.randint(radius // 2, shape.height)

    while total_steps_remaining > 0:
        explanation = spatial_search(
            img_array, prediction_func, targets, radius, radius_eta, pixel_ranking, r, c, mask_value, chunk_size
        )
        if explanation is None:
            local = obj_func(neighbours((shape.width, shape.height, shape.channels), radius, r, c, pixel_ranking, 0.0))
            attempts = step_size * 2
            new_r, new_c = random_steps(r, c, step_size, shape.width, shape.height)
            near = obj_func(
                neighbours((shape.width, shape.height, shape.channels), radius, new_r, new_c, pixel_ranking, 0.0)
            )
            while local < near and attempts > 0:
                near = obj_func(
                    neighbours((shape.width, shape.height, shape.channels), radius, new_r, new_c, pixel_ranking, 0.0)
                )
                new_r, new_c = random_steps(r, c, step_size, shape.width, shape.height)
                attempts -= 1
            if attempts == 0:
                r, c = np.random.randint(radius // 2, shape.width), np.random.randint(radius // 2, shape.height)
            else:
                r = new_r
                c = new_c
        else:
            return explanation

        total_steps_remaining -= 1
    return None
