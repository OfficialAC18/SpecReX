#!/usr/bin/env python
'''
Adopted Code from Private Code Repository:
Multiple Different Black Box Explanations for Image Classifiers by Chockler et al.
https://arxiv.org/abs/2309.14309

'''


import logging
import sys
import os


def set_log_level(i: int, logger):
    if i == 0:
        logger.setLevel(logging.CRITICAL)
    elif i == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logger = logging.getLogger("ReX")
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.FATAL)
