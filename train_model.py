"""
train_model.py

Train a MobileNetV2-based classifier on the `women/` dataset.

Usage:
    python train_model.py

Outputs:
    - models/fashion_model.h5 (best model)
    - models/labels.txt (class names)
    - models/training_history.png (accuracy/loss)
    - models/confusion_matrix.png
    - models/classification_report.txt

This file is beginner-friendly and commented for a semester project.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report
import itertools

# ---------------------
# Configuration
# ---------------------
DATA_DIR = "women"  # dataset directory with subfolders per class
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 123
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fashion_model.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")
HISTORY_PLOT = os.path.join(MODEL_DIR, "training_history.png")
CM_PLOT = os.path.join(MODEL_DIR, "confusion_matrix.png")
REPORT_PATH = os.path.join(MODEL_DIR, "classification_report.txt")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------
# Check GPU (optional)
# ---------------------
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("GPU detected:", gpus)
else:
    print("No GPU detected, training will use CPU.")

# ---------------------
# Load datasets
# ---------------------
print("Loading datasets from disk...")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Detected classes:", class_names)

# Save labels for predict.py
with open(LABELS_PATH, "w", encoding="utf-8") as f:
    for c in class_names:
        f.write(c + "\n")

# ---------------------
# Data pipeline: augmentation and preprocessing
# ---------------------
AUTOTUNE = tf.data.AUTOTUNE

# Data augmentation (on-the-fly)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
])

def prepare(ds, shuffle=False, augment=False):
    # Resize already done by image_dataset_from_directory, but ensure type
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    # Preprocess for MobileNetV2 (scales to [-1,1])
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(buffer_size=AUTOTUNE)

train_ds = prepare(train_ds, shuffle=True, augment=True)
val_ds = prepare(val_ds)

# ---------------------
# Build model (MobileNetV2 transfer learning)
# ---------------------
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base_model.trainable = False  # freeze base for initial training

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------------
# Callbacks: EarlyStopping and ModelCheckpoint
# ---------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
]

# ---------------------
# Train model
# ---------------------
EPOCHS = 15
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# Optionally fine-tune: unfreeze and train a few epochs
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

FT_EPOCHS = 5
ft_history = model.fit(train_ds, validation_data=val_ds, epochs=FT_EPOCHS, callbacks=callbacks)

# Save final model (best model already saved by checkpoint)
model.save(MODEL_PATH)

# ---------------------
# Plot training history (accuracy & loss)
# ---------------------
def plot_history(h1, h2=None, save_path=HISTORY_PLOT):
    acc = h1.history.get("accuracy", []) + (h2.history.get("accuracy", []) if h2 else [])
    val_acc = h1.history.get("val_accuracy", []) + (h2.history.get("val_accuracy", []) if h2 else [])
    loss = h1.history.get("loss", []) + (h2.history.get("loss", []) if h2 else [])
    val_loss = h1.history.get("val_loss", []) + (h2.history.get("val_loss", []) if h2 else [])

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Train Acc")
    plt.plot(epochs, val_acc, label="Val Acc")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

plot_history(history, ft_history)

# ---------------------
# Evaluate: confusion matrix and classification report
# ---------------------
print("Evaluating on validation set and generating report...")

# Gather all validation images and labels
y_true = np.concatenate([y.numpy() for x, y in val_ds], axis=0)
preds = model.predict(val_ds)
y_pred = np.argmax(preds, axis=1)

cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_names)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

def plot_confusion_matrix(cm, classes, save_path=CM_PLOT, normalize=False):
    plt.figure(figsize=(8, 6))
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

plot_confusion_matrix(cm, class_names, normalize=False)

print("Training complete. Model and reports saved in:")
print(MODEL_PATH)
print(REPORT_PATH)
print(HISTORY_PLOT)
print(CM_PLOT)
