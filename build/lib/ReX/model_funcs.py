#!/usr/bin/env python
# from typing import List
from scipy.special import softmax
import tensorflow as tf
import numpy.typing as npt
import platform
import numpy as np
import copy
import cv2
import onnxruntime as ort
import sys

from ReX.logger import logger


class Shape:
    def __init__(self, array) -> None:
        #(batch_size, channels, length)
        try:
            _, x, y = array
        except:
            _, x, y = array.shape
        if x == 3 or x == 1:
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
    else:
        return np.zeros((shape.length, shape.channels), dtype=bool)
    
def spectra_mask_multi(spectra: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    #Return a copy of the spectra as the mask
    return copy.deepcopy(spectra)


#Default Normalization is SNV
def convert_image_generic(path, x, y, means=None, stds=None):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(x, y))
    img = tf.keras.preprocessing.image.img_to_array(img)

    img = img.transpose(2, 0, 1)
    img = img.astype("float32")

    if means is not None and stds is not None:
        logger.info("applying min-max normalization")
        norm_img_data = np.zeros(img.shape).astype("float32")
        for i in range(img.shape[0]):
            norm_img_data[i, :, :] = (img[i, :, :] / 255 - means[i]) / stds[i]  # type: ignore
        return np.expand_dims(norm_img_data, axis=0)
    else:
        img = img / 255.0  # type: ignore
        img = np.expand_dims(img, axis=0)
        return img


def prepare_image(path, shape=None, means=None, stds=None):
    if shape is None:
        return convert_image_generic(path, 0, 0, means=means, stds=stds)
    else:
        return convert_image_generic(path, shape.width, shape.height, means=means, stds=stds)


def get_onxx_prediction(mutant, top_predictions, sess, input_name):
    predictions = sess.run(None, {input_name: mutant})[0][0]
    ps = np.argsort(predictions)[-top_predictions:]
    probabilities = softmax(predictions)
    return (ps, probabilities[ps])


def get_prediction(model, img_array, verbose=0, top_predictions=1):
    """return the top prediction(s)"""
    predictions = model.predict(img_array, verbose=verbose)
    probabilities = softmax(predictions)
    ps = np.argsort(predictions)[0][-top_predictions:]
    return ps, probabilities[0][ps]


def model_load(model, compile=True):
    m = None
    if model == "mobilenet":
        m = tf.keras.applications.mobilenet.MobileNet()
    if model == "vgg19":
        m = tf.keras.applications.vgg19.VGG19()
    if model.endswith(".model") or model.endswith(".h5") or model.endswith(".hdf5"):
        m = tf.keras.models.load_model(model)
    if m is not None and compile:
        m.compile(optimizer="adam")
        return m
    else:
        sys.exit(-1)


def get_prediction_function(model, top_predictions, gpu):
    if type(model) == str:
        if model.endswith(".onnx"):
            sess_options = ort.SessionOptions()
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
        else:
            m = model_load(model)
            return (
                lambda mutant: get_prediction(m, mutant, top_predictions=top_predictions),
                Shape(m.input_shape),
            )
    else:
        logger.warning(f"did not recognise {model}, so loading mobilenet")
        # assume a default of mobilenet
        model = model_load("mobilenet")
        return (
            lambda mutant: get_prediction(model, mutant, top_predictions=top_predictions),
            model.input_shape,
        )
