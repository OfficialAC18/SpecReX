import os
import pytest
import copy
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from ReX.explanation import explanation, spectral_explanations
from ReX.config import get_all_args
from ReX.model_funcs import get_prediction_function
from ReX.visualisation import spectra_ranking_plot


#Required args
args = get_all_args(["Enter Args here"])


#Load Prediction Function
model = get_prediction_function(
    args.model,
    args.top_predictions,
    args.gpu,
    args.model_file,
    args.model_name,
    args.model_config,
    args.input_shape
)[0]

#Get the responsibility landscape
pos_ranking, _, spec_array, wn_array = explanation(args)
pos_ranking = pos_ranking/np.max(pos_ranking)

fig, axs = plt.subplots(nrows=1,
                       ncols=1,
                       figsize=(12,6))
axs.plot(pos_ranking)
fig.savefig("./test_resp")

#Args for single and multiple explainations
single_args = copy.deepcopy(args)
multiple_args = copy.deepcopy(args)

single_args.multiple = False
multiple_args.multiple = True

#create new

@pytest.mark.parametrize('args, spec_array, wn_array, prediction_func, pos_ranking',[
                            (single_args, spec_array, wn_array, model, pos_ranking),
                            (multiple_args, spec_array, wn_array, model, pos_ranking)])


def test_explanation_extraction(args, spec_array, wn_array, prediction_func, pos_ranking):
    
    #We are going to call fixed_beam_search and check how many explanations are being returned by it
    explanations = spectral_explanations(args, spec_array,
                                        wn_array, prediction_func, pos_ranking)
    
    if args.multiple == True:
        assert len(explanations) > 1

    else:
        assert len(explanations) == 1

    name, ext = os.path.splitext(args.output[0])
    out = f"{name}_{args.targets[0]}{ext}"
    #Also test if visualisation is generated for these explanation
    spectra_ranking_plot(out, spec_array, wn_array, pos_ranking, explanations)