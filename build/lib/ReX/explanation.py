#!/usr/bin/env python3
from __future__ import annotations

from tqdm import trange
from scipy.ndimage import center_of_mass

import cv2
import time

from ReX.distributions import Distribution
from ReX.multi_explanation import multi_spotlight, spotlight_search
from ReX.model_funcs import *
from ReX.logger import logger
from ReX.database import add_to_database, initialise_rex_db
from ReX.ranking import linear_search, spatial_search, Strategy
from ReX.responsibility import causal_explanation


def generate_multi_explanations(
    img_array,
    prediction_func,
    targets,
    pixel_ranking,
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
            img_array,
            prediction_func,
            targets,
            pixel_ranking,
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
    img_array,
    prediction_func,
    targets,
    pixel_ranking,
    radius,
    radius_eta,
    mask_value,
    step_size,
    strategy,
    chunk_size,
    no_expansions,
):
    if strategy == Strategy.Linear:
        return linear_search(img_array, prediction_func, targets, pixel_ranking, mask_value, chunk_size)
    if strategy == Strategy.Chunk:
        # TODO adaptive linear search
        pass
    if strategy == Strategy.Spatial:
        sort = np.argsort(pixel_ranking, axis=None)
        r, c = center_of_mass(pixel_ranking)
        rows, cols = np.unravel_index(sort, pixel_ranking.shape)
        rows = rows[::-1]
        cols = cols[::-1]
        # print((r, c), (rows[0], cols[0]))
        return spatial_search(
            img_array,
            prediction_func,
            targets,
            radius,
            radius_eta,
            pixel_ranking,
            rows[0],
            cols[0],
            mask_value,
            chunk_size,
            no_expansions=no_expansions,
        )
    if strategy == Strategy.Spotlight:
        # TODO include no expansions
        return spotlight_search(
            img_array,
            prediction_func,
            targets,
            pixel_ranking,
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


def single_or_multi(args, img_array, prediction_func, pixel_ranking):
    if args.strategy == Strategy.MultiSpotlight:
        explanations = generate_multi_explanations(
            img_array,
            prediction_func,
            args.targets,
            pixel_ranking,
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
            img_array,
            prediction_func,
            args.targets,
            pixel_ranking,
            args.spatial_radius,
            args.spatial_eta,
            args.mask_value,
            args.spotlight_step,
            args.strategy,
            args.chunk_size,
            args.no_expansions,
        )

        return expln, False


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
    pixel_ranking,
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
                pixel_ranking,
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
            pixel_ranking,
            explanations,
            time_taken,
            passing=passing,
            failing=failing,
            total_work=total,
            depth_reached=depth_reached,
            avg_box_size=avg_box_size,
        )


def explanation(args):
    prediction_func, input_shape = get_prediction_function(args.model, args.top_predictions, args.gpu)

    if args.preprocess is not None:
        logger.info("using the user-provided preprocess script %s", args.preprocess_location)
        img_array = np.array(cv2.imread(args.path))
        img_array = args.preprocess(img_array)
    else:
        # if img is already the correct size etc, then just turn it into a numpy array
        if args.processed:
            img_array = np.array(cv2.imread(args.path))
            img_array = img_array.astype("float32")
            img_array = img_array / 255.0  # type: ignore
            img_array = np.expand_dims(img_array, axis=0)
        # try to process the image in an appropriate fashion
        else:
            logger.info("preprocessing image")
            img_array = prepare_image(args.path, shape=input_shape, means=args.means, stds=args.stds)

        img_shape = Shape(img_array.shape)
        if img_shape.channels != input_shape.channels:
            img_array = img_array.transpose(0, 3, 1, 2)

        if input_shape.order != img_shape.order:
            img_array = img_array.transpose(0, 2, 3, 1)

    db = None
    if args.db is not None:
        db = initialise_rex_db(args.db)

    if args.targets is None:
        args.targets = prediction_func(img_array)[0]
    logger.info("image classified as %s", args.targets)

    img_shape = Shape(img_array.shape)

    start = time.time()
    passing: int = 0
    failing: int = 0
    depth_reached: int = 0
    avg_box_size: float = 0.0
    pixel_ranking = None

    if args.iters >= 1:
        resp_map = None
        for i in trange(args.iters):
            # for i in range(0, args.iters):
            if args.adaptive and i > args.bootstrap:
                logger.info("using adaptive sampling")
                args.distribution = Distribution.Adaptive
            invert = False
            if args.switch is not None and args.adaptive and i > args.bootstrap:
                invert = i % args.switch == 0
                if invert:
                    logger.info("inverting pixel ranking on this iteration")
            r, p, f, dr, avg_size = causal_explanation(
                i,
                img_array,
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
        pixel_ranking = resp_map if resp_map.max() == 0 else resp_map / resp_map.max()  # type: ignore
        avg_box_size /= args.iters

    explanations, is_multi = single_or_multi(args, img_array, prediction_func, pixel_ranking)
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
            pixel_ranking,
            depth_reached=depth_reached,
            avg_box_size=avg_box_size,
        )

    return pixel_ranking, explanations
