import numpy as np
from scipy.interpolate import CubicSpline
import random
import pytest

import warnings
warnings.filterwarnings("ignore")

from ReX.specaug import interpolate_mask


#Test examples for spectra and wavenumbers
spectra = np.random.randn(1,891).reshape(1,-1)
wn = np.arange(stop=891).reshape(1,-1)

#Mask for sending positions to be interpolated
mask = np.ones_like(spectra)

#Random start position and stop position for the spectra 
start = random.randint(a = 50, b = 450)
stop = random.randint(a = 500, b = 800)

#Set interpolated region to false
mask[:,start:stop] = 0

@pytest.mark.parametrize("mask, wn, spectra, method",[
    (mask, wn, spectra, "linear"),
    (mask, wn, spectra, "cubic")
])


def test_mutant_generation(mask,
                           wn,
                           spectra,
                           method):
    
    #Create a mutant array and compare to the outputs of the function
    original_values = np.where(mask == 1)
    interp_values = np.where(mask == 0)

    test_mutant = np.zeros_like(mask, dtype = 'float32')
    test_mutant[original_values] = spectra[original_values]
    
    sorted_spec_pos = np.argsort(wn[np.where(mask == 1)])


    if method == "linear":
        test_mutant[interp_values] = np.interp(wn[interp_values],
                                  wn[original_values][sorted_spec_pos],
                                  spectra[original_values][sorted_spec_pos])
    else:
        test_mutant[interp_values] = CubicSpline(wn[original_values][sorted_spec_pos],
                                                    spectra[original_values][sorted_spec_pos])(wn[interp_values])
    

    #Generate mutant from function
    mutant = interpolate_mask(mask,
                              wn,
                              spectra,
                              method)
    

    #Check if both mutants are the same
    assert np.array_equal(mutant,test_mutant)