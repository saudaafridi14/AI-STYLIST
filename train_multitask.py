import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------------
# Configuration
# ---------------------
DATASET_DIR = "dataset"
TRAIN_CSV = os.path.join(DATASET_DIR, "train_labels.csv")
VAL_CSV = os.path.join(DATASET_DIR, "val_labels.csv")
LABELS_JSON = os.path.join("models", "multitask_labels.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Slightly smaller to accommodate multiple outputs on memory
SEED = 123
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fashion_multitask_model.h5")
HISTORY_PLOT = os.path.join(MODEL_DIR, "multitask_history.png")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------
# Read Labels Metadata
# ---------------------
if not os.path.exists(LABELS_JSON):
    raise FileNotFoundError(f"Metadata file '{LABELS_JSON}' not found. Please run prepare_multitask_data.py first.")

with open(LABELS_JSON, "r", encoding="utf-8") as f:
    labels_metadata = json.load(f)

styles = labels_metadata["styles"]
patterns = labels_metadata["patterns"]
colors = labels_metadata["colors"]

num_styles = len(styles)
num_patterns = len(patterns)
num_colors = len(colors)

print(f"Loaded Metadata:")
print(f" - Styles ({num_styles}): {styles}")
print(f" - Patterns ({num_patterns}): {patterns}")
print(f" - Colors ({num_colors}): {colors}")

# ---------------------
# Data Pipeline
# ---------------------
def parse_image_and_labels(path, style, pattern, color):
    try:
        img_raw = tf.io.read_file(path)
        # Decode and set static shape
        img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        img.set_shape(IMG_SIZE + (3,))
        return img, {
            "style_output": style,
            "pattern_output": pattern,
            "color_output": color
        }
    except Exception as e:
        # Fallback to zero image on read errors
        return tf.zeros(IMG_SIZE + (3,), dtype=tf.float32), {
            "style_output": style,
            "pattern_output": pattern,
            "color_output": color
        }

def build_dataset(csv_path):
    df = pd.read_csv(csv_path)
    
    # We want absolute/relative paths from the project dir
    paths = df["image_path"].values
    style_labels = df["style_label"].values.astype(np.int32)
    pattern_labels = df["pattern_label"].values.astype(np.int32)
    color_labels = df["color_label"].values.astype(np.int32)
    
    dataset = tf.data.Dataset.from_tensor_slices((paths, style_labels, pattern_labels, color_labels))
    dataset = dataset.shuffle(buffer_size=len(df), seed=SEED)
    dataset = dataset.map(parse_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset, len(df)

print("\nLoading datasets...")
train_raw_ds, num_train = build_dataset(TRAIN_CSV)
val_raw_ds, num_val = build_dataset(VAL_CSV)

print(f"Found {num_train} training samples and {num_val} validation samples.")

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
])

def augment(img, labels):
    img = data_augmentation(img, training=True)
    return img, labels

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_raw_ds.map(augment, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds = val_raw_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# ---------------------
# Build Multi-Task Model
# ---------------------
print("\nBuilding multi-task model...")
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze base for initial training

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

# Define task heads
style_output = layers.Dense(num_styles, activation="softmax", name="style_output")(x)
pattern_output = layers.Dense(num_patterns, activation="softmax", name="pattern_output")(x)
color_output = layers.Dense(num_colors, activation="softmax", name="color_output")(x)

model = models.Model(inputs=inputs, outputs=[style_output, pattern_output, color_output])

# Compile with task-specific loss functions
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={
        "style_output": "sparse_categorical_crossentropy",
        "pattern_output": "sparse_categorical_crossentropy",
        "color_output": "sparse_categorical_crossentropy",
    },
    loss_weights={
        "style_output": 1.0,
        "pattern_output": 0.8,  # Balance weights based on task complexity
        "color_output": 0.8,
    },
    metrics={
        "style_output": "accuracy",
        "pattern_output": "accuracy",
        "color_output": "accuracy",
    },
)

model.summary()

# ---------------------
# Training
# ---------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
]

EPOCHS = 10
print(f"\nStarting Stage 1 training (backbone frozen) for {EPOCHS} epochs...")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# Optional fine-tuning stage
print("\nUnfreezing top layers of MobileNetV2 backbone for fine-tuning...")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss={
        "style_output": "sparse_categorical_crossentropy",
        "pattern_output": "sparse_categorical_crossentropy",
        "color_output": "sparse_categorical_crossentropy",
    },
    loss_weights={
        "style_output": 1.0,
        "pattern_output": 0.8,
        "color_output": 0.8,
    },
    metrics={
        "style_output": "accuracy",
        "pattern_output": "accuracy",
        "color_output": "accuracy",
    },
)

FT_EPOCHS = 4
print(f"Starting Stage 2 training (fine-tuning) for {FT_EPOCHS} epochs...")
ft_history = model.fit(train_ds, validation_data=val_ds, epochs=FT_EPOCHS, callbacks=callbacks)

# Save final model
model.save(MODEL_PATH)
print(f"\nModel training finished and saved to {MODEL_PATH}")

# ---------------------
# Plot Training History
# ---------------------
def plot_multitask_history(h1, h2, save_path):
    # Combine training curves
    epochs_range = range(1, len(h1.history["loss"]) + len(h2.history["loss"]) + 1)
    
    style_acc = h1.history["style_output_accuracy"] + h2.history["style_output_accuracy"]
    val_style_acc = h1.history["val_style_output_accuracy"] + h2.history["val_style_output_accuracy"]
    
    pattern_acc = h1.history["pattern_output_accuracy"] + h2.history["pattern_output_accuracy"]
    val_pattern_acc = h1.history["val_pattern_output_accuracy"] + h2.history["val_pattern_output_accuracy"]
    
    color_acc = h1.history["color_output_accuracy"] + h2.history["color_output_accuracy"]
    val_color_acc = h1.history["val_color_output_accuracy"] + h2.history["val_color_output_accuracy"]
    
    loss = h1.history["loss"] + h2.history["loss"]
    val_loss = h1.history["val_loss"] + h2.history["val_loss"]

    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, style_acc, label="Train Style Acc")
    plt.plot(epochs_range, val_style_acc, label="Val Style Acc")
    plt.legend()
    plt.title("Style Classification Accuracy")
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, pattern_acc, label="Train Pattern Acc")
    plt.plot(epochs_range, val_pattern_acc, label="Val Pattern Acc")
    plt.legend()
    plt.title("Pattern Classification Accuracy")
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, color_acc, label="Train Color Acc")
    plt.plot(epochs_range, val_color_acc, label="Val Color Acc")
    plt.legend()
    plt.title("Color Classification Accuracy")
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.legend()
    plt.title("Total Loss")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training history curves to {save_path}")

plot_multitask_history(history, ft_history, HISTORY_PLOT)

if __name__ == "__main__":
    print("Multi-task training pipeline complete.")
