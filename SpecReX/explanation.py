#!/usr/bin/env python3

from __future__ import annotations

from tqdm import trange

import pandas as pd
import time

from SpecReX.spectral_explanations import fixed_beam_search
from SpecReX.model_funcs import *
from SpecReX.logger import logger
from SpecReX.responsibility import causal_explanation, causal_explanation_wrapper

def spectral_explanations(args, spec_array, wn_array, prediction_func, pos_ranking):
    return fixed_beam_search(
        spec_array=spec_array,
        wn_array=wn_array,
        prediction_func=prediction_func,
        pos_ranking=pos_ranking,
        interp_method=args.interp_method,
        beam_size=args.beam_size,
        beam_engulf_window=args.beam_engulf_window,
        beam_eta=args.beam_eta,
        responsibility_similarity=args.responsibility_similarity,
        maxima_scaling_factor=args.maxima_scaling_factor,
        multiple=args.multiple,
        target_class=args.targets
    )

def summarise(points):
    pmax = points.max()

    pmax_pos = np.unravel_index(points.argmax(), points.shape)

    pm = points.mean()
    ps = points.std()

    return pmax, pmax_pos, pm, ps, np.median(points)


def explanation(args):
    prediction_func, input_shape = get_prediction_function(args.model, args.top_predictions, args.gpu, args.model_file, args.model_name, args.model_config, args.input_shape)

    if args.preprocess is not None:
        logger.info("using the user-provided preprocess script %s", args.preprocess_location)
        spec_array, wn_array = args.preprocess(args.spectra_path,args.wn_array)

        #Make sure the spectra and wavenumber are the same shape
        assert spec_array.shape == wn_array.shape, "Spectra and Wavenumber are of different shapes"

    else:
        # if spectra and wavenumber is already processed, then just turn it into a numpy array
        if args.processed:
            if 'csv' in args.spectra_path:
                spec_array = pd.read_csv(args.spectra_path, header = None).values
            else:
                spec_array = np.load(args.spectra_path)
            
            if 'csv' in args.wn_path:
                wn_array = pd.read_csv(args.wn_path, header = None).values
            else:
                wn_array =  np.load(args.wn_path)
                
            spec_array = spec_array.astype("float32")
            wn_array = spec_array.astype("float32")

            #Expanding dimension for meeting batching requirements of input
            spec_array = np.expand_dims(spec_array, axis=0)
            wn_array = np.expand_dims(wn_array, axis=0)
            spec_shape = Shape(spec_array)

        #Perform standard input processing
        else:
            logger.info("Pre-processing spectra")
            spec_array, wn_array, spec_shape = prepare_spectra_wn(args.spectra_path,
                                                                args.wn_path,
                                                                shape = input_shape,
                                                                means=args.means,
                                                                stds=args.stds)

    if args.targets is None:
        args.targets = prediction_func(spec_array)[0]
    logger.info("spectra classified as %s", args.targets)

    start = time.time()
    passing: int = 0
    failing: int = 0
    depth_reached: int = 0
    avg_box_size: float = 0.0
    pos_ranking = None

    if args.iters >= 1:
        resp_map = None
        for i in trange(args.iters):
            r, p, f, dr, avg_size = causal_explanation(i,
                                                    spec_array,
                                                    wn_array,
                                                    spec_shape,
                                                    args,
                                                    responsibility_map=resp_map,
                                                    min_work=args.min_work,
                                                    seed=args.seed,
                                                    prediction_func=prediction_func,
                                                    bounding_box=None)
            resp_map = r
            passing += p
            failing += f
            depth_reached = max(dr, depth_reached)
            avg_box_size += avg_size
        pos_ranking = resp_map if resp_map.max() == 0 else resp_map / resp_map.max()  # type: ignore
        avg_box_size /= args.iters

    explanations = spectral_explanations(args, spec_array, wn_array, prediction_func, pos_ranking)
    end = time.time()
    time_taken = end - start
    logger.info(time_taken)

    return pos_ranking, explanations, spec_array, wn_array


def explanation_wrapper(prediction_func, spec_array,
                         wn_array, spec_shape,
                        iters, distribution,
                        distribution_args, search_limit,
                        tree_depth, weighted,
                        min_box_size, interp_method,
                        min_work, total_restart_attempts, seed, 
                        bounding_box, verbose,
                        targets = None, return_mutant_iters = -1):
    
    if targets == None:
        targets = prediction_func(spec_array)[0]
    print("Spectra Classified as:", targets)

    start = time.time()
    passing : int = 0
    failing : int = 0
    depth_reached: int = 0
    avg_box_size: float = 0.0
    pos_ranking = None
    extracted_mutants = []

    assert iters >= 1, "Number of iterations need to be >= 1"
    resp_map = None
    for i in trange(iters):
        r, p, f, dr, avg_size = causal_explanation_wrapper(process = i,
                                                           spec_array = spec_array,
                                                           wn_array = wn_array, 
                                                           spec_shape = spec_shape,
                                                           distribution = distribution,
                                                           distribution_args = distribution_args,
                                                           search_limit = search_limit,
                                                           tree_depth = tree_depth,
                                                           targets = targets,
                                                           weighted = weighted,
                                                           min_box_size = min_box_size,
                                                           interp_method = interp_method,
                                                           responsibility_map = resp_map,
                                                           min_work = min_work,
                                                           total_restart_attempts = total_restart_attempts,
                                                           repeated = False,
                                                           seed = seed,
                                                           prediction_func = prediction_func,
                                                           verbose = verbose,
                                                           bounding_box = bounding_box,
                                                           return_mutant_iters=return_mutant_iters,
                                                           extracted_mutants=extracted_mutants)
        resp_map = r
        passing += p
        failing += f
        depth_reached = max(dr, depth_reached)
        avg_box_size += avg_size
    
    pos_ranking = resp_map if resp_map.max() == 0 else resp_map / resp_map.max()
    avg_box_size /= iters

    time_taken = time.time() - start
    print("Time taken:",time_taken," secs")

    return pos_ranking, targets, extracted_mutants