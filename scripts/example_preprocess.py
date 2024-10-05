"""user provided preprocess script"""

import numpy as np
import cv2


def preprocess(img_array):
    """takes in a raw array from cv2.imread() and preprocesses it,
    returning a numpy array of the correct form for the model"""
    img_array = cv2.resize(img_array, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    img_array = img_array.transpose(2, 0, 1)
    img_array = img_array.astype("float32")
    img_array = img_array / 255.0  # type: ignore
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
