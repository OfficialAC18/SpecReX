"""main entry point to ReX"""

import os

from ReX.config import get_all_args
from ReX.image_generation import produce_image, masked_image
from ReX.explanation import explanation, summarise
from ReX.logger import logger, set_log_level
from ReX.ranking import Strategy
import numpy as np

RANKING_DIR = "Path to directory where the rankings/explanations are saved"
def main():
    """main entry point to ReX cmdline tool"""
    args = get_all_args()
    set_log_level(args.verbosity, logger)

    logger.debug("running ReX with the following args:\n %s", args)

    ranking, exp, *_ = explanation(args)

    np.save(os.path.join(RANKING_DIR,"explanation.npy"),explanation)
    np.save(os.path.join(RANKING_DIR,"ranking.npy"),ranking)
    exit(-1)

    pm, pos, mean, std, median = summarise(ranking)
    logger.info(
        "max ranking value %f, at position %s," + "ranking mean %f, with std %f and median %f.",
        pm,
        pos,
        mean,
        std,
        median,
    )
    if exp is not None:
        if args.output is not None:
            name, ext = os.path.splitext(args.output[0])
            if args.strategy == Strategy.MultiSpotlight:
                for i, e in enumerate(exp):
                    if args.targets is not None:
                        out = f"{name}_{args.targets[0]}_{str(i).zfill(2)}{ext}"
                        masked_image(args.path, out, e, args.mask_value, processed=args.processed)
            else:
                if args.targets is not None:
                    out = f"{name}_{args.targets[0]}{ext}"  # type: ignore
                    masked_image(args.path, out, exp, args.mask_value, processed=args.processed)
    if args.surface is not None or args.contour is not None or args.heatmap is not None:
        produce_image(args, ranking)


if __name__ == "__main__":
    main()