import pytest
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from ReX.model_funcs import get_prediction_function

#Test Model Loading
 
PATH_ONNX = ''
TOP_PREDS = 1
PATH_PYTORCH_WEIGHTS = ''
PATH_PYTORCH_DEF = ''
PYTORCH_NAME = ''
PATH_CONFIG = ''

test_example = ''
input_shape_val = (1,1,852)

#Known test values
output_class = 0
output_probs_onnx = [1.0000000e+00, 1.4601790e-15, 4.0994313e-12]
output_probs_pytorch = [1.0000e+00, 1.4602e-15, 4.0994e-12]


@pytest.mark.parametrize("model_load, top_preds, gpu, model_file, model_name,model_config, input_shape",[
    (PATH_ONNX, TOP_PREDS, True, None, None, None, None),
    (PATH_ONNX, TOP_PREDS, False, None, None, None, None),
    (PATH_PYTORCH_WEIGHTS, TOP_PREDS, True, PATH_PYTORCH_DEF, PYTORCH_NAME, PATH_CONFIG, input_shape_val),
    (PATH_PYTORCH_WEIGHTS, TOP_PREDS, False, PATH_PYTORCH_DEF, PYTORCH_NAME, PATH_CONFIG, input_shape_val)
])

def test_get_predicition_function(model_load,
                                  top_preds,
                                  gpu,
                                  model_file,
                                  model_name,
                                  model_config,
                                  input_shape):
    #Load Model
    model, _ = get_prediction_function(model_load,
                                  top_preds,
                                  gpu,
                                  model_file,
                                  model_name,
                                  model_config,
                                  input_shape)

    #Load the test example and batch it
    example = np.load(test_example).astype(np.float32).reshape(input_shape_val)

    #If it is a ONNX path
    if model_load.endswith('.onnx'):
        onnx_pred, onnx_probs = model(example)

        #Test if the model outputs are as expected
        assert np.array_equal(output_class,onnx_pred[0]), f'Failed: Expected:{output_class}, Got:{onnx_pred}'
        assert np.array_equal(output_probs_onnx[output_class], onnx_probs[0]), f"Failed: Expected:{output_probs_onnx[output_class]}, Got:{onnx_probs}"
    
    #If it is a PyTorch path
    else:
        pytorch_pred, pytorch_probs = model(example)

        assert np.array_equal(output_class, pytorch_pred[0]), f'Failed: Expected:{output_class}, Got:{pytorch_pred}'
        assert np.array_equal(output_probs_pytorch[output_class], pytorch_probs[0]), f"Failed: Expected:{output_probs_pytorch[output_class]}, Got:{pytorch_probs}"



    