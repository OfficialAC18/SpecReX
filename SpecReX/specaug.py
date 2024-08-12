#!/usr/bin/env python
import numpy as np
import random
from scipy.signal import savgol_filter

def interpolate_mask(mask,
                    wavenumber,
                    spectra,
                    method = "linear"):
    
    '''
    Interpolate the unmasked regions of the spectra
    (Currently only supports 1D Spectra)
    
    args:
        mask - Binary mask containing positions to be interpolated (1: Interpolate, 0: Keep Original)
        wavenumbers - Wavenumbers to be interpolated (x values)
        spectra - Spectral data to be interpolated (y values)
        method - Type of interpolation to be performed (linear, cubic (splines))

    Returns:
        mutant - Interpolated array of values
    '''
    #Values to be interpolated
    interp_pos = np.where(mask == 0)

    #Values to be kept as original spectra
    spec_pos = np.where(mask == 1)

    #Create a mutant array of the shape of the mask
    mutant = np.zeros(mask.shape, dtype = 'float32')

    #Put the unmasked region back in the mask
    mutant[spec_pos] = spectra[spec_pos]

    if np.any(interp_pos):
        #Sort the wavenumber regions for the interpolation functions
        sorted_spec_pos = np.argsort(wavenumber[spec_pos])

        if method == "linear":
            interp_region = np.interp(wavenumber[interp_pos],
                                    wavenumber[spec_pos][sorted_spec_pos],
                                    spectra[spec_pos][sorted_spec_pos])
            mutant[interp_pos] = interp_region
        
        elif method == "cubic":
            from scipy.interpolate import CubicSpline
            interp_func = CubicSpline(wavenumber[spec_pos][sorted_spec_pos],
                                    spectra[spec_pos][sorted_spec_pos])
            
            interp_region = interp_func(wavenumber[interp_pos])
            mutant[interp_pos] = interp_region

    return mutant

