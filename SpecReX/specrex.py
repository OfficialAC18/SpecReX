"""main entry point to SpecReX"""
'''
Adopted Code from Private Code Repository:
Multiple Different Black Box Explanations for Image Classifiers by Chockler et al.
https://arxiv.org/abs/2309.14309

'''

import os
import sys

import numpy as np

from SpecReX.config import get_all_args
from SpecReX.visualisation import spectra_ranking_plot
from SpecReX.explanation import explanation, summarise
from SpecReX.logger import logger, set_log_level

def main():
    """main entry point to SpecReX cmdline tool"""
    args = get_all_args(sys.argv[1:])
    set_log_level(args.verbosity, logger)

    logger.debug("running SpecReX with the following args:\n %s", args)

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