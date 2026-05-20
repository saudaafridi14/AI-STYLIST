import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Short smoke training to verify pipeline
DATA_DIR = r"C:\Users\pc\OneDrive\Desktop\women"
IMG_SIZE = (224,224)
BATCH_SIZE = 16
VAL_SPLIT = 0.2
SEED = 123
MODEL_DIR = "models"
TEST_MODEL_PATH = os.path.join(MODEL_DIR, "fashion_model_test.h5")

os.makedirs(MODEL_DIR, exist_ok=True)

print('TensorFlow', tf.__version__)
print('GPU devices:', tf.config.list_physical_devices('GPU'))

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset='training',
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset='validation',
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

print('Classes:', train_ds.class_names)

# Limit dataset size for quick run (few batches)
train_ds = train_ds.take(6)
val_ds = val_ds.take(3)

AUTOTUNE = tf.data.AUTOTUNE

def prepare(ds, augment=False):
    ds = ds.map(lambda x,y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    if augment:
        aug = tf.keras.Sequential([
            layers.RandomFlip('horizontal'),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.08),
        ])
        ds = ds.map(lambda x,y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x,y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

train_ds = prepare(train_ds, augment=True)
val_ds = prepare(val_ds, augment=False)

class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR,d))])
num_classes = len(class_names)
print('Detected class count:', num_classes)

# Build small transfer model
base_model = MobileNetV2(input_shape=IMG_SIZE+(3,), include_top=False, weights='imagenet')
base_model.trainable = False
inputs = layers.Input(shape=IMG_SIZE+(3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

# Train 1 epoch
history = model.fit(train_ds, validation_data=val_ds, epochs=1)

# Save test model
model.save(TEST_MODEL_PATH)
print('Saved test model to', TEST_MODEL_PATH)

# Quick evaluate
loss, acc = model.evaluate(val_ds)
print('Validation loss, acc:', loss, acc)
