# SpecReX 
Causal Responsibility Explanations for spectral classifiers

<picture>
 <source media="(prefers-color-scheme: dark)" srcset="assets/SpecReX.png">
 <source media="(prefers-color-scheme: light)" srcset="assets/SpecReX.png">
 <img alt="ReX Logo with dinosaur" src="YOUR-DEFAULT-IMAGE">
</picture>

# Setup

The following instructions assumes ```conda```

``` bash
conda create -n srex python=3.10
conda activate srex
pip install .
```

onnxruntime-gpu can cause problems. Either install it manually or edit the pyproject.toml to read "onnxruntime >= 1.17.0" rather than 
"onnxruntime-gpu >= 1.17.0"

This should install an executable ```rex``` in your path.

# Simple Usage

``` bash
# with linear search
rex <path_to_image> --model <path_to_model> --strategy linear

# with spatial search (the default)
rex <path_to_image> --model <path_to_model> 

# to save the extracted explanation
rex <path_to_image> --model <path_to_model> --output <path_and_extension>

# to view an interactive responsibility landscape
rex <path_to_image> --model <path_to_model>  --surface 

# to save a responsibility landscape
rex <path_to_image> --model <path_to_model>  --surface <path_and_extension>

# to run multiple explanations
rex <path_to_image> --model <path_to_model> --strategy multi
```

# Database
To store all output in a sqlite database, use 

``` bash
rex <path_to_image> --model <path_to_model> -db <name_of_db_and_extension>
```

ReX will create the db if it does not already exist. It will append to any db with the given name, so be careful not to use the same database if you are
restarting an experiment.

# Config
ReX looks for the config file <rex.toml> in the current working directory and then ```$HOME/.config/rex.toml``` on unix-like systems.

If you want to use a custom location, use 

``` bash
rex <path_to_image> --model <path_to_model> --config <path_to_config>
```

# Preprocessing

ReX by default tries to make reasonable guesses for image preprocessing. If the image has already been resized appropriately for the model, then
use the processed flag

``` bash
rex <path_to_image> --model <path_to_model> --processed
```

ReX will still normalize the image and convert it into a numpy array. In the event the the model input is single channel and the image is multi-channel, then ReX will try to convert the image to greyscale. If you want to avoid this, then pass in a greyscale image. 

## Preprocess Script

If you have very specific requirements for preprocessing, you can write a standalone function, ```preprocess(array)``` which ReX will try to load dynamically and call

``` bash
rex <path_to_image> --model <path_to_model> --process_script <path_to_script.py>
```

An example is included in ```scripts/example_preprocess.py```

# SpecReX (Example)
In order to execute SpecReX, refer to the following example
``` bash
rex --spectra_filename <path_to_spectra> --wn_filename <path_to_wavenumber> --model <path_to_model> (Optional) --ranking_dir <path_to_dir> (Output) --output <filename.ext>
```
