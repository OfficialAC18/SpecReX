import os
import torch
import numpy as np
from tqdm import tqdm


from ReX.explanation import explanation
from ReX.model_funcs import get_prediction_function
from ReX.config import get_all_args

from captum.attr import GradientShap, KernelShap

#Test Model Loading
TOP_PREDS = 1
PYTORCH_NAME = ['InSilicoConv', 'InSilicoLRCN', 'InSilicoLSTM']

PATH_PYTORCH_WEIGHTS_SINGLE = ['/home/akchunya/Akchunya/MSc Thesis/Final Models/PyTorch/Single Peak Conv/checkpoint.pt', '/home/akchunya/Akchunya/MSc Thesis/Final Models/PyTorch/Single Peak LRCN/checkpoint.pt', '/home/akchunya/Akchunya/MSc Thesis/Final Models/PyTorch/Single Peak LSTM/checkpoint.pt']
PATH_PYTORCH_WEIGHTS_DOUBLE = ['/home/akchunya/Akchunya/MSc Thesis/Final Models/PyTorch/Double Peak Conv/checkpoint.pt','/home/akchunya/Akchunya/MSc Thesis/Final Models/PyTorch/Double Peak LRCN/checkpoint.pt', '/home/akchunya/Akchunya/MSc Thesis/Final Models/PyTorch/Double Peak LSTM/checkpoint.pt']
PATH_PYTORCH_WEIGHTS_COMPLEX = ['/home/akchunya/Akchunya/MSc Thesis/Final Models/PyTorch/Complex Peak Conv/checkpoint.pt', '/home/akchunya/Akchunya/MSc Thesis/Final Models/PyTorch/Complex Peak LRCN/checkpoint.pt','/home/akchunya/Akchunya/MSc Thesis/Final Models/PyTorch/Complex Peak LSTM/checkpoint.pt']

ALL_PYTORCH_WEIGHTS = [PATH_PYTORCH_WEIGHTS_SINGLE, PATH_PYTORCH_WEIGHTS_DOUBLE, PATH_PYTORCH_WEIGHTS_COMPLEX]

PATH_PYTORCH_DEF_SINGLE = ['/home/akchunya/Akchunya/MSc Thesis/Final Models/insilico_conv.py', '/home/akchunya/Akchunya/MSc Thesis/Final Models/insilico_lrcn.py', '/home/akchunya/Akchunya/MSc Thesis/Final Models/insilico_lstm.py']
PATH_PYTORCH_DEF_DOUBLE = ['/home/akchunya/Akchunya/MSc Thesis/Final Models/insilico_conv.py', '/home/akchunya/Akchunya/MSc Thesis/Final Models/insilico_lrcn.py', '/home/akchunya/Akchunya/MSc Thesis/Final Models/insilico_lstm.py']
PATH_PYTORCH_DEF_COMPLEX = ['/home/akchunya/Akchunya/MSc Thesis/Final Models/insilico_conv.py', '/home/akchunya/Akchunya/MSc Thesis/Final Models/insilico_lrcn.py', '/home/akchunya/Akchunya/MSc Thesis/Final Models/insilico_lstm.py']

ALL_PYTORCH_DEF = [PATH_PYTORCH_DEF_SINGLE, PATH_PYTORCH_DEF_DOUBLE, PATH_PYTORCH_DEF_COMPLEX]

PATH_CONFIG_SINGLE = ['/home/akchunya/Akchunya/MSc Thesis/Final Models/PyTorch/Single Peak Conv/params.json', '/home/akchunya/Akchunya/MSc Thesis/Final Models/PyTorch/Single Peak LRCN/params.json', '/home/akchunya/Akchunya/MSc Thesis/Final Models/PyTorch/Single Peak LSTM/params.json']
PATH_CONFIG_DOUBLE = ['/home/akchunya/Akchunya/MSc Thesis/Final Models/PyTorch/Double Peak Conv/params.json', '/home/akchunya/Akchunya/MSc Thesis/Final Models/PyTorch/Double Peak LRCN/params.json', '/home/akchunya/Akchunya/MSc Thesis/Final Models/PyTorch/Double Peak LSTM/params.json']
PATH_CONFIG_COMPLEX = ['/home/akchunya/Akchunya/MSc Thesis/Final Models/PyTorch/Complex Peak Conv/params.json', '/home/akchunya/Akchunya/MSc Thesis/Final Models/PyTorch/Complex Peak LRCN/params.json','/home/akchunya/Akchunya/MSc Thesis/Final Models/PyTorch/Complex Peak LSTM/params.json']

ALL_PYTORCH_CONFIG = [PATH_CONFIG_SINGLE, PATH_CONFIG_DOUBLE, PATH_CONFIG_COMPLEX]
WN_FILES = ['/home/akchunya/Akchunya/MSc Thesis/Simulated Dataset/Single_Double_peak_wn.npy', '/home/akchunya/Akchunya/MSc Thesis/Simulated Dataset/Single_Double_peak_wn.npy', '/home/akchunya/Akchunya/MSc Thesis/Simulated Dataset/Complex_peak_wn.npy']

MODEL_SHAPES = [(1,1,852), (1,1,852), (1,1,1752)]

SPECREX_RESP_DRIVE = '/home/akchunya/Akchunya/MSc Thesis/Saliency Landscapes/SpecReX (Responsibility Values)'
KSHAP_SALIENCY_DRIVE = '/home/akchunya/Akchunya/MSc Thesis/Saliency Landscapes/KernelSHAP (Feature Importance)'
GSHAP_SALIENCY_DRIVE = '/home/akchunya/Akchunya/MSc Thesis/Saliency Landscapes/GradientSHAP (Feature Importance)'

single_peak_examples = [os.path.join('/home/akchunya/Akchunya/MSc Thesis/Simulated Dataset/Single Peak',file) for file in os.listdir('/home/akchunya/Akchunya/MSc Thesis/Simulated Dataset/Single Peak')]
double_peak_examples = [os.path.join('/home/akchunya/Akchunya/MSc Thesis/Simulated Dataset/Double Peak',file) for file in os.listdir('/home/akchunya/Akchunya/MSc Thesis/Simulated Dataset/Double Peak')]
complex_peak_examples = [os.path.join('/home/akchunya/Akchunya/MSc Thesis/Simulated Dataset/Complex Peak',file) for file in os.listdir('/home/akchunya/Akchunya/MSc Thesis/Simulated Dataset/Complex Peak')]

all_examples = [single_peak_examples, double_peak_examples, complex_peak_examples]
dataset_names = ['Single','Double','Complex']

def calculate_saliency():
    for dataset_weights, dataset_def, dataset_config, dataset_shape, dataset_wn, dataset_name, examples in tqdm(zip(ALL_PYTORCH_WEIGHTS, ALL_PYTORCH_DEF, ALL_PYTORCH_CONFIG, MODEL_SHAPES, WN_FILES, dataset_names, all_examples)):
        for model_weights, model_def, model_config, model_name in zip(dataset_weights, dataset_def, dataset_config, PYTORCH_NAME):
            for spec in examples:
                # Get the args
                args = get_all_args(['--spectra_filename',
                                spec,
                                '--wn_filename',
                                dataset_wn,
                                '--model_file',
                                model_def,
                                '--model',
                                model_weights,
                                '--model_name',
                                model_name,
                                '--input_shape',
                                f'{dataset_shape[0]}', f'{dataset_shape[1]}', f'{dataset_shape[2]}',
                                '--model_config',
                                model_config])
                
                #Set GPU to true
                args.gpu = True
            
                #Load Model
                _,loaded_model,_ = get_prediction_function(args.model,
                                        args.top_predictions,
                                        args.gpu,
                                        args.model_file,
                                        args.model_name,
                                        args.model_config,
                                        args.input_shape)
                        
                #KernelSHAP
                ks = KernelShap(loaded_model)
                #GradientSHAP
                gs = GradientShap(loaded_model)

                #Load examples
                spec_arr = np.load(spec)
                
                #Get Responsibility
                pos_ranking, _, _, _ = explanation(args)
                pos_ranking = pos_ranking/np.max(pos_ranking)

                if 'class_0' in spec:
                    target = 0
                elif 'class_1' in spec:
                    target = 1
                elif 'class_2' in spec:
                    target = 2
            
                #Calculate KernelSHAP values
                ks_attr = ks.attribute(torch.from_numpy(spec_arr.reshape(1,1,-1).astype(np.float32)).cuda(),
                            target = target,
                            n_samples=2000).detach().cpu().numpy().reshape(-1)

                #Set model to train here
                loaded_model.train()
                #Calculate GradientSHAP values
                # gs_attr = gs.attribute(torch.from_numpy(spec_arr.reshape(1,1,-1).astype(np.float32)).cuda(),
                #             torch.zeros_like(torch.from_numpy(spec_arr.reshape(1,1,-1).astype(np.float32))).cuda(),
                #             target = target,
                #             n_samples=2000).detach().cpu().numpy().reshape(-1)

                #Save to drive
                if 'Conv' in model_name:
                    model_type = 'Conv'
                elif 'LRCN' in model_name:
                    model_type = 'LRCN'
                elif 'LSTM' in model_name:
                    model_type = 'LSTM'

                filename = spec.split('/')[-1][:-4]

                np.save(os.path.join(SPECREX_RESP_DRIVE,f'{dataset_name} Peak {model_type}',filename+'.npy'),pos_ranking)
                np.save(os.path.join(KSHAP_SALIENCY_DRIVE,f'{dataset_name} Peak {model_type}',filename+'.npy'),ks_attr)
                # np.save(os.path.join(GSHAP_SALIENCY_DRIVE,f'{dataset_name} Peak {model_type}',filename+'.npy'),gs_attr)



calculate_saliency()