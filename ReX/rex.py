"""main entry point to ReX"""

import os

from ReX.config import get_all_args
from ReX.image_generation import produce_image, spectra_ranking_plot
from ReX.explanation import explanation, summarise
from ReX.logger import logger, set_log_level
from ReX.ranking import Strategy
import numpy as np

def main():
    """main entry point to ReX cmdline tool"""
    args = get_all_args()
    set_log_level(args.verbosity, logger)

    logger.debug("running ReX with the following args:\n %s", args)

    ranking,*_ = explanation(args)
    if hasattr(args,'ranking_dir'):
        np.save(os.path.join(args.ranking_dir,"ranking_2_1.npy"),ranking)

    pm, pos, mean, std, median = summarise(ranking)
    logger.info(
        "max ranking value %f, at position %s," + "ranking mean %f, with std %f and median %f.",
        pm,
        pos,
        mean,
        std,
        median,
    )

    if args.output is not None:
        name, ext = os.path.splitext(args.output[0])
        if args.targets is not None:
            out = f"{name}_{args.targets[0]}{ext}"
            spectra_ranking_plot(out,np.load(args.spectra_path),np.load(args.wn_path),ranking)

    if args.surface is not None or args.contour is not None or args.heatmap is not None:
        produce_image(args, ranking)


if __name__ == "__main__":
    main()