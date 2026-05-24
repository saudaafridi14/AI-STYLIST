"""
preprocess.py

Utility functions for single-image preprocessing used by predict.py.

Functions:
    - load_and_preprocess_image(path): returns preprocessed array ready for model.predict

This module uses OpenCV for reading images and handles corrupted files safely.
"""

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)

def load_and_preprocess_image(path):
    """Read image from `path` with OpenCV, convert to RGB, resize, and preprocess for MobileNetV2.

    Returns:
        image_array: numpy array with shape (1, 224, 224, 3) or None if loading failed.
    """
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype("float32")
        # Preprocess for MobileNetV2 (scales to [-1,1])
        img = preprocess_input(img)
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img
    except Exception:
        return None
