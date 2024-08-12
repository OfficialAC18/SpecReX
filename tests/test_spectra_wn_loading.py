import pytest
import numpy as np

import warnings
warnings.filterwarnings("ignore")


from ReX.model_funcs import prepare_spectra_wn, Shape

#1) Loading File as (1,Spec Len)
#2) Loading File as (Spec Len, 1)
#3) Loading File from .npy
#4) Loading File from .csv

#

PATH_FIRST_NUMPY = ''
PATH_LAST_NUMPY = ''
PATH_FIRST_CSV = ''
PATH_LAST_CSV = ''

required_length = 852
order_first = 'first'
order_last = 'last'
shape_first = Shape((1,1,852))
shape_last = Shape((1,852,1))

#Random mean and std dev values to allow for testing the user-provided values
mean = [np.random.rand()]
std = [np.random.rand()]

@pytest.mark.parametrize("spectra_path, wn_path, shape, means, stds",[
    (PATH_FIRST_CSV, PATH_FIRST_CSV, shape_first, None, None),
    (PATH_FIRST_CSV, PATH_FIRST_CSV, shape_first, mean, std),
    (PATH_LAST_CSV, PATH_LAST_CSV, shape_last, None, None),
    (PATH_LAST_CSV, PATH_LAST_CSV, shape_last, mean, std),
    (PATH_FIRST_NUMPY, PATH_FIRST_NUMPY, shape_first, None, None),
    (PATH_FIRST_NUMPY, PATH_FIRST_NUMPY, shape_first, mean, std),
    (PATH_LAST_NUMPY, PATH_LAST_NUMPY, shape_last, None, None),
    (PATH_LAST_NUMPY, PATH_LAST_NUMPY, shape_last, mean, std),
])

def test_spectra_wn_loading(spectra_path, 
                            wn_path,
                            shape,
                            means,
                            stds):

    #Load the files
    spec_array, wn_array, _ = prepare_spectra_wn(spectra_path, 
                                            wn_path,
                                            shape,
                                            means,
                                            stds)
    
    #Check the shape of the corresponding files
    if 'first' in spectra_path:
        assert Shape(spec_array.shape).order == 'first'
        assert Shape(wn_array.shape).order == 'first'

    else:
        assert Shape(spec_array.shape).order == 'last'
        assert Shape(wn_array.shape).order == 'last'
        assert spec_array.shape == (1,spec_array.shape[1],spec_array.shape[2])
    
    #Check that the array has been batched
    assert spec_array.shape == (1,spec_array.shape[1],spec_array.shape[2])
    assert wn_array.shape == (1, wn_array.shape[1],wn_array.shape[2])

    #Check both have the same shape
    assert spec_array.shape == wn_array.shape

    #Check that truncation has worked
    assert Shape(spec_array).length == 852
    assert Shape(wn_array).length == 852
