#!/usr/bin/env python

"""
calculate causal responsibility
"""
import os
from enum import Enum
from typing import List
import cv2

import numpy as np
from anytree.cachedsearch import find

from ReX.distributions import Distribution

from ReX.model_funcs import get_prediction_function, Shape, negative_mask_multi

from ReX.specaug import interpolate_mask

from ReX.box import average_box_length, initialise_tree, build_tree

from ReX.logger import logger

CAUSAL = Enum("CAUSAL", ["Responsibility"])

_combinations = [
    [
        0,
    ],
    [
        1,
    ],
    [
        2,
    ],
    [
        3,
    ],
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 2],
    [1, 3],
    [2, 3],
    [0, 1, 2],
    [0, 1, 3],
    [0, 2, 3],
    [1, 2, 3],
]


def apply_combination(mask, children, i):
    """apply combination of parts to mask"""
    sets = [children[j] for j in _combinations[i] if j < len(children)]
    for s in sets:
        s.apply_to_mask(mask)
    return sets


def subbox(tree, name):
    """find the current active subbox and spawn children"""
    to_split = find(tree, lambda node: node.name == name)
    if to_split is not None:
        return to_split.children
    return []


def set_held(tree, explanation, held) -> None:
    """Retain the regions we are holding in the rest of the partition"""
    for b_name in held:
        box = find(tree, lambda node: node.name == b_name)
        if box is not None:
            box.apply_to_mask(explanation)


def responsibility(parts, weights):
    """calculate responsibility"""
    output = np.zeros(4, dtype=np.float32)
    for w, part in enumerate(parts):
        k = len(part)
        for p in part:
            i = np.uint(p[-1])
            if weights == []:
                output[i] += 1 / k
            else:
                output[i] += weights[w] * 1 / k

    return output


def causal_explanation(
    process,
    spec_array,
    wn_array,
    spec_shape,
    args,
    responsibility_map=None,
    min_work=0.2,
    total_restart_attempts=5,
    repeated=False,
    seed=None,
    prediction_func=None,
    invert=True,
    bounding_box=None,  # of the form [row_start, row_stop]
):
    """calculate causal responsiblity"""
    if isinstance(prediction_func, str):
        prediction_func, _ = get_prediction_function(prediction_func, args.top_predictions, args.weighted)

    if seed is not None:
        if repeated:
            new = seed + process + total_restart_attempts * 100
            np.random.seed(new)
            seed = new
        else:
            new = process + seed
            np.random.seed(new)
            seed = new
        logger.info("random seed = %d", seed)

    if responsibility_map is None:
        responsibility_map = np.zeros((spec_shape.length), dtype=np.float32)

    if bounding_box is not None:
        if len(bounding_box) != 2:
            logger.error("bounding_box should be a list of length 2, not %d", len(bounding_box))
            raise IndexError
        tree = initialise_tree(
            bounding_box[1],
            args.distribution,
            args.distribution_args,
            r_start=bounding_box[0],
        )
    else:
        tree = initialise_tree(spec_shape.length, args.distribution, args.distribution_args)

    if args.distribution == Distribution.Adaptive and np.sum(responsibility_map) > 0.0:
        build_tree(tree, args.tree_depth, args.min_box_size, pixel_ranking=responsibility_map, invert=invert)
    else:
        build_tree(tree, args.tree_depth, args.min_box_size, invert=True)

    total_work = 0
    total_passing = 0
    total_failing = 0

    depth_reached = 0
    iters = 0
    queue = [tree.name]

    box_lengths = {}

    flag = True
    while flag:
        logger.info(
            "main causal loop for process %d: iter = %d, depth reached = %d, " "total work so far = %d",
            process,
            iters,
            depth_reached,
            total_passing + total_failing,
        )

        if len(queue) == 0:
            logger.info("%d quitting here, as no smaller box passes", process)
            flag = False
            break

        #Why is this there, won't we just not construct the tree to greater than tree_depth?
        if depth_reached > args.tree_depth:
            logger.info("sufficient depth reached after %d iterations for process %d", iters, process)
            flag = False
            break

        if total_passing + total_failing > args.search_limit:
            logger.info("total work exceed: %d for process %d", total_passing + total_failing, process)
            flag = False
            break

        mutants = []
        partitions = []
        passing_partitions = []
        passing_mutants = []
        held = []

        for job in queue:
            mask = negative_mask_multi(spec_shape)
            for processing in job:
                held = [p for p in job if p not in [processing]]
                children = subbox(tree, processing)
                #Testing if an adaptive box size works better
                # args.min_box_size = np.mean([child.length() for child in children])
                children = list(filter(lambda child: child.length() >= args.min_box_size , children)) #Can we avoid creating them in the first place?

                if len(children) == 0:
                    break

                for box in children:
                    box_lengths[box.name] = box.length()

                #This is the magic sauce, this is where the major changes have to be done
                #This needs to be optimized
                for i in range(14):
                    #We need to save this seperately, rather than make it common with set held
                    partition = apply_combination(mask, children, i)
                    set_held(tree, mask, held)

                    #This is in order to make sure bad mutant fails
                    if np.any(mask):
                        #Now, Create the required mutant
                        mutant = interpolate_mask(mask,wn_array[0,:,:],spec_array[0,:,:])

                        #Append the mutant to the mutant list
                        mutants.append(mutant)
                        
                        #Add required partitions to the list    
                        partitions.append(partition)
                    
                    mask[:] = False

        if len(mutants) == 0:
            break

        total_work += len(mutants)

        #Parallelize this
        #Create an arg value when intializing the prediction funtion
        predictions = [prediction_func(np.expand_dims(mutant,axis = 0)) for mutant in mutants]  # type: ignore #Parallelize this, push as batch
        weights = None
        l = list(zip(*predictions))

        if args.weighted:
            predictions = list(l[0])
            weights = list(l[1])
        else:
            predictions = list(l[0])

        resp_weights = []
        for i, pred in enumerate(predictions):
            #This is to check if the predictions match what is required
            if len(np.intersect1d(args.targets, pred)) > 0: #and l[1][i] > 0.90:
                passing_mutants.append(mutants[i])
                pp = [child.name for child in partitions[i]]
                if len(pp) > 0:
                    passing_partitions.append(pp)
                    if weights is not None:
                        if len(weights[i]) == 1:
                            resp_weights.append(weights[i][0])
                        else:
                            resp_weights.append(weights[i][0][pred])
                total_passing += 1
            else:
                total_failing += 1

        rp = responsibility(passing_partitions, resp_weights)

        if np.sum(rp) == 0.0:
            break
        children = np.unique(np.hstack(passing_partitions))
        for box in children:
            box = find(tree, lambda node: node.name == box)
            if box is not None:
                depth_reached = max(depth_reached, box.depth)
                add = rp[int(box.name[-1])]
                responsibility_map[box.row_start : box.row_stop] += add

        areas = [np.sum([box_lengths[j] for j in job]) for job in passing_partitions]
        take = np.argsort(areas)
        queue = [passing_partitions[i] for i in take[:1]] #Checking what happens when we take multiple regions
        iters += 1

    if total_work < (args.search_limit * min_work) and total_restart_attempts > 0:
        if total_restart_attempts == 2:
            logger.warning("restaring iteration %d (1 attempt remaining) " "as minimun work not achieved", process)
        else:
            logger.warning(
                "restarting iteration %d (%d attempts remaining) " "as minimum work of %d is not achieved.",
                process,
                total_restart_attempts - 1,
                args.search_limit * min_work,
            )

        return causal_explanation(
            process,
            spec_array,
            args,
            total_restart_attempts=total_restart_attempts - 1,
            repeated=True,
            seed=seed,
            min_work=min_work,
            prediction_func=prediction_func,
        )

    logger.info(
        "iteration %d = TOTAL PASSING: %d, TOTAL FAILING: %d " "MAX TREE DEPTH: %d, AVERAGE BOX LENGTH: %f",
        process,
        total_passing,
        total_failing,
        depth_reached,
        average_box_length(tree, depth_reached),
    )

    return (responsibility_map, total_passing, total_failing, depth_reached, average_box_length(tree, depth_reached))
