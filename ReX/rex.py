"""main entry point to ReX"""

import os
import sys

from ReX.config import get_all_args
from ReX.visualisation import spectra_ranking_plot
from ReX.explanation import explanation, summarise
from ReX.logger import logger, set_log_level

def main():
    """main entry point to ReX cmdline tool"""
    args = get_all_args(sys.argv[1:])
    set_log_level(args.verbosity, logger)

    logger.debug("running ReX with the following args:\n %s", args)

    ranking, explanations ,spec_array, wn_array = explanation(args)

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
            spectra_ranking_plot(out,spec_array,wn_array, ranking, explanations)

if __name__ == "__main__":
    main()