#!/usr/bin/env python
'''
Modified Code from Private Code Repository:
Multiple Different Black Box Explanations for Image Classifiers by Chockler et al.
https://arxiv.org/abs/2309.14309

'''

"""
calculate causal responsibility
"""
from enum import Enum
from typing import List

import numpy as np
import os
from anytree.cachedsearch import find

from SpecReX.distributions import Distribution

from SpecReX.model_funcs import get_prediction_function, Shape, negative_mask_multi

from SpecReX.specaug import interpolate_mask

from SpecReX.box import average_box_length, initialise_tree, build_tree

from SpecReX.logger import logger

CAUSAL = Enum("CAUSAL", ["Responsibility"])
MUTANT_PATH = "/home/akchunya/Akchunya/Raman Spectra Paper/SpecReX/mutants and responsibility"

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


def set_held(tree, explanation, held, spec_shape) -> None:
    """Retain the regions we are holding in the rest of the partition"""
    for b_name in held:
        box = find(tree, lambda node: node.name == b_name)
        if box is not None:
            if spec_shape.order == "first":
                explanation[:,box.row_start:box.row_stop] = box.mutant[:,box.row_start:box.row_stop]
            else:
                explanation[box.row_start:box.row_stop,:] = box.mutant[box.row_start:box.row_stop,:]


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

    build_tree(tree, args.tree_depth, args.min_box_size)

    total_work = 0
    total_passing = 0
    total_failing = 0

    depth_reached = 0
    iters = 0
    queue = [tree.name]

    box_lengths = {}
    save_mutants = 2

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
        failing_partitions = []
        passing_mutants = []
        held = []

        for job in queue:
            mask = negative_mask_multi(spec_shape)
            for processing in job:
                held = [p for p in job if p not in [processing]]
                children = subbox(tree, processing)
                
                #Get the processing node
                processing = find(tree, lambda node: node.name == processing)

                #Inherit the interpolated regions from the parent, only interpolate the new regions
                start = processing.row_start
                stop = processing.row_stop
                input_array = processing.mutant if processing.mutant is not None else spec_array[0,:,:]


                children = list(filter(lambda child: child.length() >= args.min_box_size , children)) #Can we avoid creating them in the first place?

                if len(children) == 0:
                    break

                for box in children:
                    box_lengths[box.name] = box.length()

                for i in range(14):
                    #We need to save this seperately, rather than make it common with set held
                    partition = apply_combination(mask, children, i)
                    
                    #Do not re-interpolate the regions that have already been interpolated
                    if spec_shape.order == 'first':
                        mask[:,0:start+1] = 1
                        mask[:,stop:] = 1
                    else:
                        mask[0:start+1,:] = 1
                        mask[stop:,:] = 1

                    #Condition allows to skip examples no regions are to be interpolated
                    if np.any(mask):
                        #Now, Create the required mutant
                        mutant = interpolate_mask(mask,wn_array[0,:,:], input_array, method = args.interp_method)

                        #Append the mutant to the mutant list
                        mutants.append(mutant)
                        
                        #Set the static region from the held partitions
                        set_held(tree, mutant, held, spec_shape)

                        #Add required partitions to the list    
                        partitions.append(partition)
                    
                    #If length of partition is 1, then save it to the node
                    if len(partition) == 1:
                        find(tree, lambda node: node.name == partition[0].name).mutant = mutant

                    mask[:] = False

        if len(mutants) == 0:
            break

        total_work += len(mutants)

        # #Save the mutants
        # if save_mutants > 0:
        #     for idx, mutant in enumerate(mutants):
        #         np.save(os.path.join(MUTANT_PATH,f"mutant_{idx}.npy"),mutant)
            
        #     save_mutants -= 1

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
            if len(np.intersect1d(args.targets, pred)) > 0:
                passing_mutants.append(mutants[i])
                pp = [child.name for child in partitions[i]]
                subset_exists = False
                #Check if a subset of the same exists
                for passed_part in passing_partitions:
                    if set(passed_part) <= set(pp):
                        subset_exists = True
                        break

                if len(pp) > 0 and not subset_exists:
                    passing_partitions.append(pp)
                    if weights is not None:
                        if len(weights[i]) == 1:
                            resp_weights.append(weights[i][0])
                        else:
                            resp_weights.append(weights[i][0][pred])
                total_passing += 1
            else:
                total_failing += 1
                fp = [child.name for child in partitions[i]]
                if len(fp) > 0:
                    failing_partitions.append(fp)

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
        queue = [passing_partitions[i] for i in take[:1]]
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


def causal_explanation_wrapper(
    process,
    spec_array,
    wn_array,
    spec_shape,
    distribution,
    distribution_args,
    search_limit,
    tree_depth,
    targets,
    weighted,
    min_box_size, # of the form [row_start, row_stop]
    interp_method,
    responsibility_map=None,
    min_work=0.2,
    total_restart_attempts=5,
    repeated=False,
    seed=None,
    prediction_func=None,
    bounding_box=None,
    verbose = False,
    extracted_mutants = [],
    return_mutant_iters = -1,
):
    """calculate causal responsiblity (wrapper function)"""
    
    if seed is not None:
        if repeated:
            new = seed + process + total_restart_attempts * 100
            np.random.seed(new)
            seed = new
        else:
            new = process + seed
            np.random.seed(new)
            seed = new
        
        if verbose:
            print("random seed = ", seed)

    if responsibility_map is None:
        responsibility_map = np.zeros((spec_shape.length), dtype=np.float32)

    if bounding_box is not None:
        if len(bounding_box) != 2:
            print("bounding_box should be a list of length 2, not ", len(bounding_box))
            raise IndexError
        tree = initialise_tree(
            bounding_box[1],
            distribution,
            distribution_args,
            r_start=bounding_box[0],
        )
    else:
        tree = initialise_tree(spec_shape.length, distribution, distribution_args)

    build_tree(tree, tree_depth, min_box_size)

    total_work = 0
    total_passing = 0
    total_failing = 0

    depth_reached = 0
    iters = 0
    queue = [tree.name]

    box_lengths = {}

    flag = True
    while flag:
        if verbose:
            print(
                "main causal loop for process %d: iter = %d, depth reached = %d, " "total work so far = %d",
                process,
                iters,
                depth_reached,
                total_passing + total_failing,
            )

            if len(queue) == 0:
                print("%d quitting here, as no smaller box passes", process)
                flag = False
                break

            if depth_reached > tree_depth:
                print("sufficient depth reached after %d iterations for process %d", iters, process)
                flag = False
                break

            if total_passing + total_failing > search_limit:
                print("total work exceed: %d for process %d", total_passing + total_failing, process)
                flag = False
                break

        mutants = []
        partitions = []
        passing_partitions = []
        failing_partitions = []
        passing_mutants = []
        held = []

        for job in queue:
            mask = negative_mask_multi(spec_shape)
            for processing in job:
                held = [p for p in job if p not in [processing]]
                children = subbox(tree, processing)
                
                #Get the processing node
                processing = find(tree, lambda node: node.name == processing)

                #Inherit the interpolated regions from the parent, only interpolate the new regions
                start = processing.row_start
                stop = processing.row_stop
                input_array = processing.mutant if processing.mutant is not None else spec_array[0,:,:]


                children = list(filter(lambda child: child.length() >= min_box_size , children)) #Can we avoid creating them in the first place?

                if len(children) == 0:
                    break

                for box in children:
                    box_lengths[box.name] = box.length()

                for i in range(14):
                    #We need to save this seperately, rather than make it common with set held
                    partition = apply_combination(mask, children, i)
                    
                    #Do not re-interpolate the regions that have already been interpolated
                    if spec_shape.order == 'first':
                        mask[:,0:start+1] = 1
                        mask[:,stop:] = 1
                    else:
                        mask[0:start+1,:] = 1
                        mask[stop:,:] = 1

                    #Condition allows to skip examples no regions are to be interpolated
                    if np.any(mask):
                        #Now, Create the required mutant
                        mutant = interpolate_mask(mask,wn_array[0,:,:], input_array, method = interp_method)

                        #Append the mutant to the mutant list
                        mutants.append(mutant)
                        
                        #Set the static region from the held partitions
                        set_held(tree, mutant, held, spec_shape)

                        #Add required partitions to the list    
                        partitions.append(partition)
                    
                    #If length of partition is 1, then save it to the node
                    if len(partition) == 1:
                        find(tree, lambda node: node.name == partition[0].name).mutant = mutant

                    mask[:] = False

        if len(mutants) == 0:
            break

        total_work += len(mutants)

        if iters in return_mutant_iters:
            for mutant in mutants:
                extracted_mutants.append(mutant)


        #Parallelize this
        #Create an arg value when intializing the prediction funtion
        predictions = [prediction_func(np.expand_dims(mutant,axis = 0)) for mutant in mutants]  # type: ignore #Parallelize this, push as batch
        weights = None
        l = list(zip(*predictions))

        if weighted:
            predictions = list(l[0])
            weights = list(l[1])
        else:
            predictions = list(l[0])

        resp_weights = []
        for i, pred in enumerate(predictions):
            #This is to check if the predictions match what is required
            if len(np.intersect1d(targets, pred)) > 0:
                passing_mutants.append(mutants[i])
                pp = [child.name for child in partitions[i]]
                subset_exists = False
                #Check if a subset of the same exists
                for passed_part in passing_partitions:
                    if set(passed_part) <= set(pp):
                        subset_exists = True
                        break

                if len(pp) > 0 and not subset_exists:
                    passing_partitions.append(pp)
                    if weights is not None:
                        if len(weights[i]) == 1:
                            resp_weights.append(weights[i][0])
                        else:
                            resp_weights.append(weights[i][0][pred])
                total_passing += 1
            else:
                total_failing += 1
                fp = [child.name for child in partitions[i]]
                if len(fp) > 0:
                    failing_partitions.append(fp)

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
        queue = [passing_partitions[i] for i in take[:1]]
        iters += 1

    if total_work < (search_limit * min_work) and total_restart_attempts > 0:
        if verbose:
            if total_restart_attempts == 2:
                print(f"restaring iteration {process} (1 attempt remaining) as minimun work not achieved")
            else:
                print(f"restarting iteration {process} ({total_restart_attempts - 1} attempts remaining) as minimum work of {search_limit * min_work} is not achieved.")
            
        return causal_explanation_wrapper(
            process,
            spec_array,
            wn_array,
            spec_shape,
            distribution,
            distribution_args,
            search_limit,
            tree_depth,
            targets,
            weighted,
            min_box_size,
            interp_method,
            responsibility_map=None,
            min_work=0.2,
            total_restart_attempts=total_restart_attempts - 1,
            repeated=True,
            seed=None,
            prediction_func=prediction_func,
            bounding_box=None,
            return_mutant_iters=return_mutant_iters,
            extracted_mutants=extracted_mutants)

    if verbose:
        print("iteration %d = TOTAL PASSING: %d, TOTAL FAILING: %d " "MAX TREE DEPTH: %d, AVERAGE BOX LENGTH: %f",
            process,
            total_passing,
            total_failing,
            depth_reached,
            average_box_length(tree, depth_reached),
        )

    return (responsibility_map, total_passing, total_failing, depth_reached, average_box_length(tree, depth_reached))