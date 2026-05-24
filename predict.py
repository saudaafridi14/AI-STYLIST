"""
predict.py

Usage:
    python predict.py /path/to/image.jpg

This script loads the trained model `models/fashion_model.h5`, reads an image
with OpenCV, preprocesses it, runs prediction, and displays the image with the
predicted class and confidence using matplotlib.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from preprocess import load_and_preprocess_image

# Paths
MODEL_PATH = os.path.join("models", "fashion_model.h5")
LABELS_PATH = os.path.join("models", "labels.txt")

def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py /path/to/image.jpg")
        return

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print("Image not found:", img_path)
        return

    # Load model and labels
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print("Trained model or labels not found. Run train_model.py first.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    labels = load_labels(LABELS_PATH)

    # Preprocess image (OpenCV)
    img = load_and_preprocess_image(img_path)
    if img is None:
        print("Failed to load or preprocess image.")
        return

    # Predict
    preds = model.predict(img)
    idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds)) * 100.0
    predicted_label = labels[idx]

    # Output
    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {confidence:.1f}%")

    # Display image with result
    orig = cv2.imread(img_path)
    if orig is None:
        print("Warning: could not read image for display")
        return
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(orig)
    plt.axis("off")
    plt.title(f"Prediction: {predicted_label} ({confidence:.1f}%)")
    plt.show()

if __name__ == "__main__":
    main()
