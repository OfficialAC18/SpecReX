#!/usr/bin/env python
import numpy as np
import random
from functools import partial
from scipy.signal import savgol_filter


try:
    import tomllib
except ModuleNotFoundError:
    import toml as tomllib

#Combinatorial Augmentation Strategey
'''
1. Load from a toml file,
    a. The set of augmentation methods to use (and their hyperparameters)
    b. Number of augments to generate
    c. Similarity Threshold
2. Select a random integer from [0, num_augments]

For 1 to rand_int
    3. Randomly sample an augment method from the set, and apply to original waveform

4. Compare Similarity b/w the Augmented example and the Original waveform
5. Store example to drive if it meets/exceeds the thereshold
'''

def wavenumber_shift(wavenumber):
    # Using a triangular distribution
    # The 'mode' argument is set to 0 to make values near 0 more likely
    shift = int(np.random.triangular(-6, 0, 6))
    return np.roll(wavenumber, shift)

def split_and_interpolate(wavenumber,
                            spectra,
                            r_start,
                            r_lim,
                            method = "linear",
                          ):
    '''
    Split the spectra at a given location and interpolate using the input method
    (Currently only supports 1D Spectra)
    
    args:
        wavenumber - Wavenumber data to be used for interpolation (x values)
        spectra - Spectral data to be interpolated (y values)
        r_start - Begining of the region to be interpolated
        r_lim - End of the region to be interpolated
        method - Type of interpolation to be performed (linear, cubic (splines))
    '''
    if method == "linear":
        interp_region = np.interp(wavenumber[r_start:r_lim],
                                  np.concatenate((wavenumber[0:r_start],wavenumber[r_lim:])),
                                  np.concatenate((spectra[0:r_start],spectra[r_lim:])))
     
        spectra[r_start:r_lim] = interp_region
    
    elif method == "cubic":
        #Select a region randomly from the spectra to slice and interpolate it using cubic splines
        from scipy.interpolate import CubicSpline
        interp_func = CubicSpline(np.concatenate((wavenumber[0:r_start],wavenumber[r_lim:])),
                                  np.concatenate((spectra[0:r_start],spectra[r_lim:])))
        
        interp_region = interp_func(wavenumber[r_start:r_lim])
        spectra[r_start:r_lim] = interp_region
    
    return spectra

def interpolate_mask(mask,
                    wavenumber,
                    spectra,
                    method = "linear"):
    
    '''
    Interpolate the unmasked regions of the spectra
    (Currently only supports 1D Spectra)
    
    args:
        mask - Binary mask containing positions to be interpolated (1: Interpolate, 0: Keep Original)
        Wavenumbers - Wavenumbers to be interpolated (x values)
        spectra - Spectral data to be interpolated (y values)
        method - Type of interpolation to be performed (linear, cubic (splines))

    Returns:
        mutant - Interpolated array of values
    '''
    #Values to be interpolated
    interp_pos = np.where(mask == True)

    #Values to be kept as original spectra
    spec_pos = np.where(mask == False)

    #Create a mutant array of the shape of the mask
    mutant = np.zeros(mask.shape, dtype = float)

    #Put the unmasked region back in the mask
    mutant[spec_pos] = spectra[spec_pos]

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

def interp_func(mask,
                wavenumber,
                spectra,
                method = "linear"):
    
    '''
    Create interpolation functions to interpolate unmasked regions of the spectra
    (Currently only supports 1D Spectra)
    
    args:
        mask - Binary mask containing positions to be interpolated (1: Interpolate, 0: Keep Original)
        Wavenumbers - Wavenumbers to be interpolated (x values)
        spectra - Spectral data to be interpolated (y values)
        method - Type of interpolation to be performed (linear, cubic (splines))

    Returns:
        interp_func - Interpolation function to use for calculating interpolated value
    '''

    #Values to be kept as original spectra
    spec_pos = np.where(mask == 0)

    #Sort the wavenumber regions for the interpolation functions
    sorted_spec_pos = np.argsort(wavenumber[spec_pos])

    if method == "linear":
        interp_func = partial(np.interp,xp = wavenumber[spec_pos][sorted_spec_pos],
                               fp = spectra[spec_pos][sorted_spec_pos])
    
    elif method == "cubic":
        from scipy.interpolate import CubicSpline
        interp_func = CubicSpline(wavenumber[spec_pos][sorted_spec_pos],
                                  spectra[spec_pos][sorted_spec_pos])

    return interp_func


def poisson_noise(wavenumber, spectra, Imax = 1000):
    '''
    Add Poisson noise to the spectra
    
    args:
        wavenumber - Input wavenumber for the spectra
        spectra - Spectral data
        Imax - Noise limit, lower value, higher noise
    
    '''
    assert np.min(spectra) >=0, "Spectrum contains negative numbers, which cannot be used for generating Poisson noise"
    wavenumber_diff = wavenumber[1] - wavenumber[0]
    
    #This converts from Photon Count to Electron count
    intensity = spectra*wavenumber_diff

    rng = np.random.default_rng()
    noise = rng.poisson(intensity/(np.max(intensity)/Imax))
    return noise * (np.max(intensity)/Imax) / wavenumber_diff


# Ask Nathan for input on this, what's the use, how 
# def control_point_shifting(wavenumber, spectra, num_control_points = 10, shift_amount = 5, scale_factor = 1, window_length = 51):
#     '''
#     Fits a new curve to a spectrum by shifting certain control points

#     Args:
#         wavenumber - Input wavenumber for the spectra
#         spectra - Spectral data.
#         num_control_points - Number of control points to use for fitting,
#         shift_amount - The amount by which to shift the control points.
#         scale_factor - Amount the scale the fitted curve by.
#         window_lenght - The size of data subsets to use for Savitzky-Golay Filters
#     Results:
#         The fitted spectrum
    
#     '''
#     from scipy.signal import savgol_filter

#     assert len(spectra) == len(wavenumber), f"Length mismatch between spectra ({len(spectra)} and wavenumber ({len(wavenumber)}))"

#     #Savitzky-Golay Spectrum Smoothing (May replace with https://pubs.acs.org/doi/epdf/10.1021/acsmeasuresciau.1c00054)
#     smooth_intensities = savgol_filter(spectra, window_length=51, )


def savitzky_golay_noise(wavenumber, spectra, window_length = 51, polyorder = 1, scaling_rate = 0.6):
    '''
    Perform Savitzky-Golay filtering on the spectra, then add the difference between smoothened and original signal
    
    args:
        wavenumber - Input wavenumber for the spectra
        spectra - Spectral data.
        window_length  - The length of the sliding window to use for fitting the polynomial
        polyorder - The degree of the polynomial to be fit
        scaling_rate - The scaling factor to be used for adding the noise
    '''

    #Apply the filtering on the spectra using the Savitsky-Golay filter
    smoothed_intensities = savgol_filter(spectra,
                                         window_length=window_length,
                                         polyorder=polyorder)




def spectra_truncation(wavenumber, spectra, start = 400, end = 1800):
    '''
    Function to truncate spectra within a specified region, default to the fingerprint region
    Inputs:
        wavenumber - an array (vector) of the wavenumbers
        spectra - an array (matrix) of spectra (spectra in the rows)
        start - start of wavenumber region to be included
        end - end of wavenumber region to be included
    Outputs:
        truncated spectra
        truncated wavenumber
    '''

    assert start < end, "Start of region must be lower than end of region"

    #Find the closest wavenumbers that match the provided input
    lower_bound = min(wavenumber, key = lambda x:abs(x - start))
    upper_bound = max(wavenumber, key = lambda x:abs(x - end))

    #Find index of the aforementioned regions
    min_wavenumber = np.squeeze(np.where(wavenumber == lower_bound))
    max_wavenumber = np.squeeze(np.where(wavenumber == upper_bound))

    #Truncate spectra and wavenumbers accordingly
    truncated_spectra = spectra[min_wavenumber:max_wavenumber]
    truncated_wavenumber = wavenumber[min_wavenumber:max_wavenumber]

    #Make sure spectra and wavenumber are of the same length
    assert len(truncated_spectra) == len(truncated_wavenumber)

    return truncated_wavenumber, truncated_spectra




def combinatorial_augment(file:str = "./comb_aug.toml") :
    with open(file,"rb") as f:
        data = tomllib.load(f)

    ind = np.argsort(wavenumber)
    wavenumber = np.take_along_axis(wavenumber,ind)
    spectra = np.take_along_axis(spectra,ind)
    
  
# if __name__ == "__main__":
#     combinatorial_augment()