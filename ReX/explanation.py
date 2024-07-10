#!/usr/bin/env python3
from __future__ import annotations

from tqdm import trange
from scipy.ndimage import center_of_mass

import pandas as pd
import time

from ReX.distributions import Distribution
from ReX.multi_explanation import multi_spotlight, spotlight_search
from ReX.spectral_explanations import fixed_beam_search
from ReX.model_funcs import *
from ReX.logger import logger
from ReX.database import add_to_database, initialise_rex_db
from ReX.ranking import linear_search, Strategy
from ReX.responsibility import causal_explanation


def generate_multi_explanations(
    spec_array,
    prediction_func,
    targets,
    pos_ranking,
    mask_value,
    initial_size,
    size_eta,
    spotlight_step_size,
    chunk_size,
    no_spotlights,
    spotlight=True,
):
    if spotlight:
        return multi_spotlight(
            spec_array,
            prediction_func,
            targets,
            pos_ranking,
            mask_value,
            initial_size,
            size_eta,
            spotlight_step_size,
            no_spotlights,
            chunk_size,
        )

    logger.warning("not implemented")
    return NotImplemented

def generate_explanation(
    spec_array,
    wn_array,
    prediction_func,
    targets,
    pos_ranking,
    radius,
    radius_eta,
    mask_value,
    step_size,
    strategy,
    chunk_size,
    no_expansions,
):
    if strategy == Strategy.Linear:
        return linear_search(spec_array, prediction_func, targets, pos_ranking, mask_value, chunk_size)
    if strategy == Strategy.Chunk:
        # TODO adaptive linear search
        pass
    if strategy == Strategy.Spatial:
        sort = np.argsort(pos_ranking, axis=None)
        r = center_of_mass(pos_ranking)
        rows = np.unravel_index(sort, pos_ranking.shape)
        rows = rows[::-1]
        print(r, rows[0][0])
        return spatial_search_spectra(
            spec_array,
            wn_array,
            prediction_func,
            targets,
            radius,
            radius_eta,
            pos_ranking,
            rows[0][0],
            mask_value,
            chunk_size,
            no_expansions=no_expansions,
        )
    if strategy == Strategy.Spotlight:
        # TODO include no expansions
        return spotlight_search(
            spec_array,
            prediction_func,
            targets,
            pos_ranking,
            mask_value,
            radius,
            radius_eta,
            step_size,
            np.mean,
            chunk_size,
        )
    raise NotImplementedError


def summarise(pixels):
    pmax = pixels.max()

    pmax_pos = np.unravel_index(pixels.argmax(), pixels.shape)

    pm = pixels.mean()
    ps = pixels.std()

    return pmax, pmax_pos, pm, ps, np.median(pixels)


def single_or_multi(args, spec_array, wn_array, prediction_func, pos_ranking):
    if args.strategy == Strategy.MultiSpotlight:
        explanations = generate_multi_explanations(
            spec_array,
            prediction_func,
            args.targets,
            pos_ranking,
            args.mask_value,
            args.spotlight_size,
            args.spotlight_eta,
            args.spotlight_step,
            args.chunk_size,
            no_spotlights=args.spotlights,
        )

        return explanations, True
    else:
        expln = generate_explanation(
            spec_array,
            wn_array,
            prediction_func,
            args.targets,
            pos_ranking,
            args.spatial_radius,
            args.spatial_eta,
            args.mask_value,
            args.spotlight_step,
            args.strategy,
            args.chunk_size,
            args.no_expansions,
        )

        return expln, False


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
        max_beams=args.max_beams,
        target_class=args.targets
    )

def update_db(
    db,
    m: bool,
    args,
    passing: int,
    failing: int,
    total: int,
    target,
    time_taken,
    explanations,
    pos_ranking,
    depth_reached=0,
    avg_box_size=0.0,
):
    logger.info("updating database")
    if m:
        for i, expln in tqdm(enumerate(explanations)):
            add_to_database(
                db,
                args,
                target,
                pos_ranking,
                expln,
                time_taken,
                passing=passing,
                failing=failing,
                total_work=total,
                multi=True,
                multi_no=i,
                depth_reached=depth_reached,
                avg_box_size=avg_box_size,
            )
    else:
        add_to_database(
            db,
            args,
            target,
            pos_ranking,
            explanations,
            time_taken,
            passing=passing,
            failing=failing,
            total_work=total,
            depth_reached=depth_reached,
            avg_box_size=avg_box_size,
        )


def explanation(args):
    prediction_func, input_shape = get_prediction_function(args.model, args.top_predictions, args.gpu) #Put a test inside this function to figure out if it can be parallezed

    if args.preprocess is not None:
        logger.info("using the user-provided preprocess script %s", args.preprocess_location)
        spec_array, wn_array = args.preprocess(args.spectra_path,args.wn_array)

        #Make sure the spectra and wavenumber are the same shape
        assert spec_array.shape == wn_array.shape, "Spectra and Wavenumber are of different shapes"

    else:
        # if spectra and wavenumber is already the correct size etc, then just turn it into a numpy array
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

            #Will add the required Nomalization once confirmed with Nathan
            # spec_array = spec_array / 255.0  # type: ignore
            spec_array = np.expand_dims(spec_array, axis=0)
            wn_array = np.expand_dims(wn_array, axis=0)

        # try to process the image in an appropriate fashion
        else:
            logger.info("preprocessing spectra")
            spec_array, wn_array = prepare_spectra_wn(args.spectra_path,
                                                   args.wn_path,
                                                   shape = input_shape,
                                                   means=args.means,
                                                   stds=args.stds)

        #Not needed really
        spec_shape = Shape(spec_array.shape)

        if spec_shape.channels != input_shape.channels:
            spec_array = spec_array.transpose(0, 2, 1)
            wn_array = wn_array.transpose(0,2,1)
            spec_shape = Shape(spec_array.shape)
        
        #This is most likely not needed
        if input_shape.order != spec_shape.order:
            spec_array = spec_array.transpose(0, 2, 1)
            wn_array = wn_array.transpose(0,2,1)
            spec_shape = Shape(spec_array.shape)

    db = None
    if args.db is not None:
        db = initialise_rex_db(args.db)

    if args.targets is None:
        #PlaceHolder for now
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
            if args.adaptive and i > args.bootstrap:
                logger.info("using adaptive sampling")
                args.distribution = Distribution.Adaptive
            invert = False
            if args.switch is not None and args.adaptive and i > args.bootstrap:
                invert = i % args.switch == 0
                if invert:
                    logger.info("inverting position ranking on this iteration")
            r, p, f, dr, avg_size = causal_explanation(
                i,
                spec_array,
                wn_array,
                spec_shape,
                args,
                responsibility_map=resp_map,
                min_work=args.min_work,
                seed=args.seed,
                prediction_func=prediction_func,
                invert=invert,
                bounding_box=None,
            )
            resp_map = r
            passing += p
            failing += f
            depth_reached = max(dr, depth_reached)
            avg_box_size += avg_size
        pos_ranking = resp_map if resp_map.max() == 0 else resp_map / resp_map.max()  # type: ignore
        avg_box_size /= args.iters

    # explanations, is_multi = single_or_multi(args, spec_array, wn_array, prediction_func, pos_ranking)
    explanations = spectral_explanations(args, spec_array, wn_array, prediction_func, pos_ranking)
    is_multi = False
    end = time.time()
    time_taken = end - start
    logger.info(time_taken)

    if db is not None:
        update_db(
            db,
            is_multi,
            args,
            passing,
            failing,
            passing + failing,
            str(args.targets),
            time_taken,
            explanations,
            pos_ranking,
            depth_reached=depth_reached,
            avg_box_size=avg_box_size,
        )

    return pos_ranking, explanations, spec_array, wn_array
