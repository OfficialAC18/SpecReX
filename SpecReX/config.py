#!/usr/bin/env python3
'''
Modified Code from Private Code Repository:
Multiple Different Black Box Explanations for Image Classifiers by Chockler et al.
https://arxiv.org/abs/2309.14309

'''



"""config management"""

from typing import List, Tuple
import sys
import csv
import argparse
import os
from os.path import exists, expanduser
import importlib.util
import numpy as np
from numpy.typing import NDArray

try:
    import tomllib as toml  # type: ignore
except ImportError:
    import toml


from SpecReX.distributions import Distribution, str2distribution
from SpecReX.responsibility import CAUSAL


class Args:
    """args argument object"""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-few-public-methods
    def __init__(self) -> None:
        self.config_location = None
        # input file
        self.spectra_path = None
        self.wn_path = None 
        self.model = None
        self.model_file = None
        self.model_name = None
        self.input_shape = None
        self.model_config = None
        self.db: None | str = None
        # gpu support
        self.gpu: bool = False
        # for reproducability
        self.seed: None | int = None
        self.preprocess = None
        self.preprocess_location = None
        self.processed = False
        # image_dims for non-standard model
        self.image_dims = None
        # min-max normalization
        self.means = None
        self.stds = None
        # verbosity
        self.verbosity = 0
        # save explanation to output
        self.output = None
        # explanation production strategy
        self.strategy: None 
        self.chunk_size = 1
        # beams args
        self.beam_size: int = 0
        self.beam_eta: int = 0
        self.beam_engulf_window: int = 0
        self.responsibility_similarity: float = 0.0
        self.maxima_scaling_factor: float = 0.0
        self.multiple: bool = True
        self.interp_method: str = None

        self.no_expansions = 0

    def __repr__(self) -> str:
        return (
            f"Args <file: {self.spectra_path}, model: {self.model}, image_dims: {self.image_dims}, gpu: {self.gpu}, "
            + f"output_file: {self.output},"
            + f"means: {self.means}, stds: {self.stds}, "
            + f"explanation_strategy: {self.strategy}, "
            + f"chunk size: {self.chunk_size},"
            + f"seed: {self.seed}, db: {self.db}, "
            + f"preprocess: {self.preprocess}, verbosity: {self.verbosity}, "
            + f"no_expansions: {self.no_expansions}, "
        )


class CausalArgs(Args):
    """Creates a causal args object"""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-few-public-methods

    def __init__(self) -> None:
        super().__init__()
        self.config_location = None
        self.type = CAUSAL
        self.tree_depth: int = 0
        self.search_limit: int = 0
        self.mask_value: int | Tuple = 0
        self.min_box_size: int = 0
        self.max_box_size: int = 0
        self.top_predictions: int = 0
        self.data_location: None | str = None
        self.targets: None | NDArray = None
        self.distribution: None | Distribution = None
        self.distribution_args: None | List = None
        self.weighted: bool = False
        self.cpus = 1
        self.iters = 1
        self.min_work = 0.0
        self.bootstrap = 2

    def __repr__(self) -> str:
        return (
            "Causal Args <"
            + Args.__repr__(self)
            + f"mask_value: {self.mask_value}, "
            + f"top_predictions: {self.top_predictions}, targets: {self.targets}, "
            + f"tree_depth: {self.tree_depth}, search_limit: {self.search_limit}, "
            + f"min_box_size: {self.min_box_size}, max_box_size: {self.max_box_size}, weighted: {self.weighted}, "
            + f"data_locations: {self.data_location}, distribution: {self.distribution}, "
            + f"distribution_args: {self.distribution_args}, cpus: {self.cpus}, iterations: {self.iters}>"
        )


def get_config_file(path):
    """parses toml file into dictionary"""
    file = open(path,'rb')
    file_args = toml.load(file)#, _dict=dict)
    return file_args


def cmdargs(args):
    """parses command line flags"""
    parser = argparse.ArgumentParser(
        prog="SpecReX",
        description="Explaining Spectral AI through causal reasoning",
    )
    parser.add_argument("--spectra_filename", type=str, required=True, help="spectra to be processed, assumes that file is 1 channel (could be channel-first or channel-last)")
    parser.add_argument("--wn_filename", type=str, required=True, help = "corresponding wavenumber for the provided input spectra, Shape of the wavenumber must be identical")
    parser.add_argument(
        "--output", nargs=1, type=str, help="save explanation to <output>. Requires a cv2 compatible file extension"
    )
    parser.add_argument("-c", "--config", type=str, help="config file to use for rex")
    parser.add_argument("--processed", action="store_true", help="don't perform any processing with rex itself")

    parser.add_argument("--process_script", type=str, help="preprocessing script")

    parser.add_argument("-v", "--verbose", action="count", default=0, help="verbosity level, either -v or -vv")
    parser.add_argument("--targets", nargs="+", type=int, help="optional label(s) to use as ground truth")
    parser.add_argument("--model", type=str, help="model, must be tensorflow, onnx or PyTorch compatible")
    parser.add_argument("--model_file", type=str, help="Definition file for the model (PyTorch)")
    parser.add_argument("--model_name", type=str, help="Name of model in config file (PyTorch)")
    parser.add_argument("--input_shape", nargs="*", type=int, default=[1,852], help="Input shape of the model (For PyTorch)")
    parser.add_argument("--model_config", type=str, help="Model configs (if any), for PyTorch model (should be a csv file)")
    parser.add_argument("--dims", nargs=2, type=int, help="image dimensions for resizing")
    parser.add_argument(
        "--strategy", "-s", type=str, help="explanation strategy, one of < multi | spatial | linear | spotlight >"
    )
    parser.add_argument(
        "--database", "-db", type=str, help="store output in sqlite database <DATABASE>, creating db if necessary"
    )

    args = parser.parse_args(args)
    return args


def key_or_default(d, key, default):
    """gets value from dictionary or returns default"""
    try:
        return d[key]
    except KeyError:
        return default


def try_dict(args, dict_name):
    """tries to get dictionary from toml file"""
    try:
        return args[dict_name]
    except KeyError:
        return {}



def shared_args(cmd_args, args):
    """parses shared args"""
    if cmd_args.config is not None:
        args.config_location = cmd_args.config
    if cmd_args.model is not None:
        args.model = cmd_args.model
    if cmd_args.model_file is not None:
        args.model_file = cmd_args.model_file
    if cmd_args.model_name is not None:
        args.model_name = cmd_args.model_name
    if cmd_args.input_shape is not None:
        args.input_shape = cmd_args.input_shape
    if cmd_args.model_config is not None:
        args.model_config = cmd_args.model_config
    if cmd_args.targets is not None:
        args.targets = np.array(cmd_args.targets)
    if cmd_args.output is not None:
        args.output = cmd_args.output
    if cmd_args.verbose > 0:
        args.verbosity = cmd_args.verbose
    args.processed = cmd_args.processed


def get_all_args(args,path=None):
    """parses all arguments from config file and command line"""
    cmd_args = cmdargs(args)

    path = None
    if cmd_args.config is not None:
        path = cmd_args.config
    else:
        conf_home = expanduser("~/.config/rex.toml")
        # search in current directory first
        if exists("rex.toml"):
            print("using rex.toml in current working directory")
            path = "rex.toml"
        # fallback on $HOME/.config on linux/macos
        elif exists(conf_home):
            print(f"using config in {conf_home}")
            path = conf_home
        else:
            print("no config found")
            sys.exit(-1)

    config_file_args = get_config_file(path)

    explain_dict = try_dict(config_file_args, "explanation")
    rex_dict = try_dict(config_file_args, "rex")
    spectral_dict = try_dict(explain_dict, "spectral")

    args = None

    # CAUSAL
    fa = config_file_args["causal"]
    args = CausalArgs()
    args.config_location = path
    args.tree_depth = key_or_default(fa, "tree_depth", 5)
    args.search_limit = key_or_default(fa, "search_limit", 200)
    args.min_work = key_or_default(fa, "min_work", 0.0)
    args.weighted = key_or_default(fa, "weighted", False)

    args.cpus = key_or_default(fa, "cpus", 1)
    args.iters = key_or_default(fa, "iters", 1)
    args.min_box_size = key_or_default(fa, "min_size", 10)
    args.max_box_size = key_or_default(fa, "max_size", args.min_box_size + 10)

    if cmd_args.process_script is not None:
        try:
            name, _ = os.path.splitext(cmd_args.process_script)
            spec = importlib.util.spec_from_file_location(name, cmd_args.process_script)
            preprocess = importlib.util.module_from_spec(spec)  # type: ignore
            spec.loader.exec_module(preprocess)  # type: ignore
            args.preprocess = preprocess.preprocess
            args.preprocess_location = cmd_args.process_script
        except ImportError:
            pass

    try:
        dist = fa["distribution"]
        d = dist["distribution"]
        args.distribution = str2distribution(d)
        args.distribution_args = key_or_default(dist, "dist_args", [])
    except KeyError:
        pass

    if args.distribution == Distribution.Uniform:
        args.distribution_args = None

    args.spectra_path = cmd_args.spectra_filename
    args.wn_path = cmd_args.wn_filename

    args.means = key_or_default(rex_dict, "means", None)
    args.stds = key_or_default(rex_dict, "stds", None)

    shared_args(cmd_args, args)

    # spectral args
    args.beam_size = key_or_default(spectral_dict, "beam_size", 20)
    args.beam_eta = key_or_default(spectral_dict, "beam_eta", 1)
    args.beam_engulf_window = key_or_default(spectral_dict, "beam_engulf_window", 10)
    args.responsibility_similarity = key_or_default(spectral_dict, "responsibility_similarity", 0.9)
    args.maxima_scaling_factor = key_or_default(spectral_dict, "scale_factor", 0.5)
    args.multiple = key_or_default(spectral_dict, "multiple", True)
    args.interp_method = key_or_default(spectral_dict, "interp_method", "linear")

    args.chunk_size = key_or_default(explain_dict, "chunk", args.min_box_size)

    # rex args
    args.mask_value = key_or_default(rex_dict, "mask_value", 0)
    args.top_predictions = key_or_default(rex_dict, "top_predictions", 1)
    args.seed = key_or_default(rex_dict, "seed", None)
    args.gpu = key_or_default(rex_dict, "gpu", False)

    return args
