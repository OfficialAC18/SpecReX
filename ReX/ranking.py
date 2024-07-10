#!/usr/bin/env python

"""explanation extraction techniques"""

from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
from enum import Enum

from ReX.model_funcs import Shape, negative_mask_multi
from ReX.specaug import interpolate_mask
from ReX.logger import logger

Strategy = Enum("Strategy", ["Linear", "Chunk", "Spatial", "Spotlight", "MultiSpotlight", "FixedBeam"])


def linear_search(img_array, prediction_func, targets, pixel_ranking, mask_value, chunk_size, shape=None):
    logger.info("starting linear search with chunk size %d", chunk_size)
    if shape is None:
        shape = Shape(img_array.shape)

    levels = []
    for i, r in enumerate(np.nditer(pixel_ranking)):
        levels.append((r, np.unravel_index(i, pixel_ranking.shape)))

    levels: List[Tuple] = sorted(levels)
    levels.reverse()

    mask = negative_mask_multi(shape)

    for i in range(0, len(levels), chunk_size):
        chunk = levels[i : i + chunk_size]
        for _, loc in chunk:
            if shape.order == "first":
                mask[loc[0], loc[1]] = True #Has been Changed for SpecReX
            else:
                mask[loc[1], loc[0]] = True #Has been changed for SpecReX
        m = simple_prediction(prediction_func, mask, targets, img_array, mask_value)
        if m is not None:
            return m

    return None


def linear_search_binary(img_array, prediction_func, targets, pixel_ranking, mask_value):
    logger.info("starting linear search")
    shape = Shape(img_array.shape)
    levels = np.flip(np.unique(pixel_ranking))
    mask = negative_mask_multi(shape)

    start = 0
    stop = len(levels)

    passing_mask = None
    while start < stop:
        mid = (start + stop) // 2
        for i in range(0, mid):
            pixels = np.where(pixel_ranking == levels[i])
            if shape.order == "first":
                # if rgb:
                mask[:, pixels[0], pixels[1]] = True
            else:
                mask[pixels[0], pixels[1], :] = True
        predictions = prediction_func(np.where(mask, img_array, mask_value))
        inter = np.intersect1d(targets, predictions[0])
        if len(inter) > 0:
            stop = mid
            passing_mask = np.copy(mask)
            mask[:] = False
        else:
            start = mid + 1

    if passing_mask is not None:
        mask = passing_mask
        return mask

    return None


def neighbours(shape, radius: int, row_number: int, column_number: int, pixel_ranking: NDArray[np.float32], val: float):
    nm = np.zeros(shape, dtype=bool)
    for i in range(row_number - 1 - radius, row_number + radius):
            if (
                i >= 0
                and i < pixel_ranking.shape[0]
                and j >= 0
                and j < pixel_ranking.shape[1]
                and pixel_ranking[i, j] >= val
            ):
                nm[i, j] = True
    return nm

def neighbours_spectra(shape, radius: int, row_number: int, pos_ranking: NDArray[np.float32], val: float):
    nm = np.zeros(shape, dtype=bool)
    for i in range(row_number - 1 - radius, row_number + radius):
            if (
                i >= 0
                and i < pos_ranking.shape[0]
                and pos_ranking[i] >= val
            ):
                nm[i] = True
    return nm


def simple_prediction(prediction_func, mask, targets, img_array, mask_value):
    im_sh = Shape(img_array)
    # TODO remove code duplication
    if im_sh.order == "first" and isinstance(mask_value, List):
        im = img_array.transpose(0, 2, 3, 1)
        m = mask.transpose(1, 2, 0)
        temp = np.where(m, im, mask_value)
        temp = temp.transpose(0, 3, 1, 2).astype("float32")
    else:
        temp = np.where(mask, img_array, mask_value)

    predictions = prediction_func(temp)[0]
    inter = np.intersect1d(targets, predictions)
    if len(inter) > 0:
        return mask
    return None

def simple_prediction_spectra(prediction_func, mask, targets, spec_array, wn_array):
    mutant = interpolate_mask(mask,wn_array[0,:,:],spec_array[0,:,:])
    predictions = prediction_func(np.expand_dims(mutant,axis = 0))[0]
    inter = np.intersect1d(targets, predictions)
    if len(inter) > 0:
        return mask
    return None
    


def ablate(explanation, prediction_func, targets, img_array, pixel_ranking, mask_value, chunk_size):
    if len(explanation.shape) == 3:
        x, y, _ = explanation.shape
        if (x, y) == pixel_ranking.shape:
            masked_pixel_ranking = np.where(explanation[:, :, 0], pixel_ranking, 0)
        else:
            masked_pixel_ranking = np.where(explanation[0, :, :], pixel_ranking, 0)
    else:
        masked_pixel_ranking = np.where(explanation, pixel_ranking, 0)
    return linear_search(img_array, prediction_func, targets, masked_pixel_ranking, mask_value, chunk_size)


def spatial_search(
    img_array,
    prediction_func,
    targets,
    radius,
    radius_eta,
    pixel_ranking,
    r,
    c,
    mask_value,
    chunk_size,
    no_expansions=10,
):
    """performs a spatial search over responsibility landscape <pixel_ranking>"""
    shape = Shape(img_array.shape)

    mask = neighbours((shape.width, shape.height, shape.channels), radius, r, c, pixel_ranking, 0.0)
    if shape.order == "first":
        mask = mask.transpose((2, 0, 1))
    logger.info(
        "performing spatial search from coordinates (%d, %d), " + "given a radius of %d and defined between %f and %f.",
        r,
        c,
        radius,
        np.min(pixel_ranking),
        np.max(pixel_ranking),
    )

    explanation = simple_prediction(prediction_func, mask, targets, img_array, mask_value)
    if explanation is None:
        for _ in range(no_expansions):
            radius = int(radius * (1 + radius_eta))
            mask = neighbours((shape.width, shape.height, shape.channels), radius, r, c, pixel_ranking, 0.0)
            if shape.order == "first":
                mask = mask.transpose((2, 0, 1))
            explanation = simple_prediction(prediction_func, mask, targets, img_array, mask_value)
            if explanation is not None:
                logger.info(f"explanation found at {(r, c)} with {radius}")
                return ablate(explanation, prediction_func, targets, img_array, pixel_ranking, mask_value, chunk_size)
    else:
        red = ablate(explanation, prediction_func, targets, img_array, pixel_ranking, mask_value, chunk_size)
        return red if red is not None else explanation

    return None

def spatial_search_spectra(
    spec_array,
    wn_array,
    prediction_func,
    targets,
    radius,
    radius_eta,
    pos_ranking,
    r,
    mask_value,
    chunk_size,
    no_expansions=10,
):
    """performs a spatial search over responsibility landscape <pos_ranking>"""
    shape = Shape(spec_array.shape)

    mask = neighbours_spectra((shape.length, shape.channels), radius, r, pos_ranking, 0.0)
    if shape.order == "first":
        mask = mask.transpose((1, 0))
    logger.info(
        "performing spatial search from coordinates (%d, %d), " + "given a radius of %d and defined between %f and %f.",
        r,
        radius,
        np.min(pos_ranking),
        np.max(pos_ranking),
    )

    explanation = simple_prediction_spectra(prediction_func, mask, targets, spec_array, wn_array)
    if explanation is None:
        for _ in range(no_expansions):
            radius = int(radius * (1 + radius_eta))
            mask = neighbours_spectra((shape.length, shape.channels), radius, r, pos_ranking, 0.0)
            if shape.order == "first":
                mask = mask.transpose((1,0))
            explanation = simple_prediction_spectra(prediction_func, mask, targets, spec_array, wn_array) #Instead of wn_array, this was previously mask_value, investigate
            if explanation is not None:
                logger.info(f"explanation found at {(r)} with {radius}")
                return ablate(explanation, prediction_func, targets, spec_array, pos_ranking, mask_value, chunk_size)
    else:
        red = ablate(explanation, prediction_func, targets, spec_array, pos_ranking, mask_value, chunk_size)
        return red if red is not None else explanation

    return None
