#!/usr/bin/env python
'''
Modified Code from Private Code Repository:
Multiple Different Black Box Explanations for Image Classifiers by Chockler et al.
https://arxiv.org/abs/2309.14309

'''


from scipy.special import softmax
import numpy.typing as npt
import platform
import numpy as np
import pandas as pd
import onnxruntime as ort
import torch
import json

from SpecReX.logger import logger


class Shape:
    def __init__(self, array) -> None:
        #(batch_size, channels, length)
        try:
            _, x, y = array
        except:
            _, x, y = array.shape

        if x == 2 or x == 1:
            self.channels = x
            self.length = y
            self.order = "first"
        else:
            self.channels = y
            self.length = x
            self.order = "last"

    def __repr__(self):
        return f"{self.length} x {self.channels}: {self.order}"


def negative_mask_multi(shape: Shape):
    if shape.order == "first":
        return np.zeros((shape.channels, shape.length), dtype=bool)
    return np.zeros((shape.length, shape.channels), dtype=bool)
    
#Default Normalization: SNV
def convert_spec_wn_generic(spectra_path, wn_path, input_length, order, means = None, stds=None):
    #Read Wavenumber and Spectra
    #Read inputs of shape (1, length) or (length, 1) or (length,)
    if 'csv' in spectra_path:
        spec_array = pd.read_csv(spectra_path, header = None).values
    else:
        spec_array = np.load(spectra_path)
    
    if 'csv' in wn_path:
        wn_array = pd.read_csv(wn_path, header = None).values
    else:
        wn_array =  np.load(wn_path)

    spec_array = spec_array.astype('float32')
    wn_array = wn_array.astype('float32')

    if len(spec_array.shape) == 1:
        if order == 'first':
            spec_array = spec_array.reshape(1,-1)
            wn_array = wn_array.reshape(1,-1)
        else:
            spec_array = spec_array.reshape(-1,1)
            wn_array = wn_array.reshape(-1,1)


    assert len(spec_array.shape) == 2, f"Expected (1,{spec_array.shape[-1]}) or ({spec_array.shape[-1]},1), Got {spec_array.shape}" 
    assert spec_array.shape == wn_array.shape, f"Mismatch in shape between spectra {spec_array.shape} and wavenumber {wn_array.shape}"

    if Shape(np.expand_dims(spec_array,axis=0)).order != order:
        spec_array = spec_array.transpose(1,0)
        wn_array = wn_array.transpose(1,0)

    assert Shape(np.expand_dims(spec_array,axis=0)).length >= input_length, "SpecReX cannot handle cases where the input is smaller than the model's input, please provide a custom processing script"
    assert Shape(np.expand_dims(wn_array,axis=0)).length >= input_length, "SpecReX cannot handle cases where the input is smaller than the model's input, please provide a custom processing script"

    #To fit to a specfic shape, the current strategy is truncation to the shape (We can't handle shapes larger than the spectra as of now)
    if input_length is not None:
        if order == 'first':
            if input_length <= spec_array.shape[1] and input_length <= wn_array.shape[1]:
                spec_array = spec_array[:,:input_length]
                wn_array = wn_array[:,:input_length]
        else:
            if input_length <= spec_array.shape[0] and input_length <= wn_array.shape[0]:
                spec_array = spec_array[:input_length,:]
                wn_array = wn_array[:input_length,:]

    if means is not None and stds is not None:
        logger.info("applying SNV normalization using provided values")

        #Make sure means is the same size as number of channels
        if order == 'first':
            assert len(means) == spec_array.shape[0], "The provided means is greater than the number of channels"
            assert len(stds) == spec_array.shape[0], "The provided stds is greater than the number of channels" 
        
            for i in range(len(means)):
                spec_array[i,:] = spec_array[i,:] - means[i]
                spec_array[i,:] = spec_array[i,:]/stds[i]
        
        if order == 'last':
            assert len(means) == spec_array.shape[-1], "The provided means is greater than the number of channels"
            assert len(stds) == spec_array.shape[-1], "The provided stds is greater than the number of channels" 
        
            for i in range(len(means)):
                spec_array[:,i] = spec_array[:,i] - means[i]
                spec_array[:,i] = spec_array[:,i]/stds[i]

    else:
        logger.info("applying SNV normalization using calculated values")
        if order == 'first':
            for i in range(spec_array.shape[0]):
                mean = np.mean(spec_array[i,:])
                std = np.std(spec_array[i,:])
                spec_array[i,:] -= mean
                spec_array[i,:] /= std
        else:
            for i in range(spec_array.shape[1]):
                mean = np.mean(spec_array[:,i])
                std = np.std(spec_array[:,i])
                spec_array[:,i] -= mean
                spec_array[:,i] /= std


        
    return np.expand_dims(spec_array,axis = 0), np.expand_dims(wn_array, axis = 0), Shape(np.expand_dims(spec_array, axis = 0))
    
   
def prepare_spectra_wn(spectra_path, wn_path, shape=None, means=None, stds=None):
    if shape is None:
        return convert_spec_wn_generic(spectra_path, wn_path, None, shape.order, means=means, stds=stds)
    else:
        return convert_spec_wn_generic(spectra_path, wn_path, shape.length, shape.order, means=means, stds=stds)

def pred_fn_wrapper(model, top_predictions):
    if isinstance(model,ort.InferenceSession):
        return lambda mutant: get_onxx_prediction(mutant, top_predictions, model, model.get_inputs()[0].name)
    elif isinstance(model, torch.nn.Module):
        return lambda mutant: get_prediction_pytorch(model, mutant, top_predictions)

def get_onxx_prediction(mutant, top_predictions, sess, input_name):
    predictions = sess.run(None, {input_name: mutant})[0][0]
    ps = np.argsort(predictions)[-top_predictions:]
    probabilities = softmax(predictions)
    return (ps, probabilities[ps])

def get_prediction_pytorch(model, input, top_predictions=1):
    with torch.no_grad():
        if next(model.parameters()).is_cuda:
            input_data = torch.from_numpy(input).to(device='cuda')
        else:
            input_data = torch.from_numpy(input)
        
        predictions = model(input_data)

    predictions = predictions.detach().cpu().numpy()
    probabilities = softmax(predictions)
    ps = np.argsort(predictions)[0][-top_predictions:]
    return ps, probabilities[0][ps]

def get_prediction_function(model, top_predictions, gpu, model_file = None, model_name = None, model_config = None, input_shape = None):
    if type(model) == str:
        if model.endswith(".onnx"):
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 0
            if gpu:
                logger.info("using gpu for onnx inference session")
                if platform.uname().system == "Darwin":
                    providers = ["CoreMLExecutionProvider"]
                else:
                    providers = [("CUDAExecutionProvider", {"enable_cuda_graph": False})]
                sess = ort.InferenceSession(model, sess_options=sess_options, providers=providers)  # type: ignore
            else:
                logger.info("using cpu for onnx inference session")
                providers = ["CPUExecutionProvider"]
                sess = ort.InferenceSession(model, sess_options=sess_options, providers=providers)
            input_name = sess.get_inputs()[0].name
            shape = sess.get_inputs()[0].shape
            logger.info(f"model shape {shape}")
            return lambda mutant: get_onxx_prediction(mutant, top_predictions, sess, input_name), Shape(shape)
        elif model.endswith('.pth') or model.endswith('.pt'):
            assert model_file is not None, "You need to pass a model file for PyTorch"
            assert model_name is not None, "You need to pass the name of the model in the model file for PyTorch"
            assert input_shape is not None, "You need to provide the shape of your input for PyTorch models"

            #Loading Model from Model file
            import importlib.util
            import sys
            spec = importlib.util.spec_from_file_location('module_name',model_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules['module_name'] = module
            spec.loader.exec_module(module)
            loaded_model = getattr(module, model_name)

            if model_config is not None:
                assert model_config.endswith('.json'), "Model config should be a JSON file"
                model_config = open(model_config)
                model_config = json.load(model_config)
                loaded_model = loaded_model(**model_config)
            else:
                loaded_model = loaded_model()
            
            if gpu:
                logger.info("using GPU for PyTorch inference session")
                loaded_model.to('cuda')
            
            #Load model weights
            loaded_model.load_state_dict(torch.load(model)[0])

            #Set model to eval
            loaded_model.eval()

            return (lambda mutant: get_prediction_pytorch(loaded_model, mutant, top_predictions=top_predictions), Shape(np.array(input_shape)))
