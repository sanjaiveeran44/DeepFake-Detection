import os
import glob
import re
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models

# ==========================================================
# 🧭 PATH SETUP
# ==========================================================
BASE_DIR = os.path.join(os.getcwd(), "dataset")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==========================================================
# ⚙️ IMAGE CROPPING FUNCTION
# ==========================================================
def crop_center(img):
    """Crop 25% from left/right and 15% from top/bottom"""
    h, w = img.shape[0], img.shape[1]
    top = int(0.15 * h)
    bottom = int(0.85 * h)
    left = int(0.25 * w)
    right = int(0.75 * w)
    return img[top:bottom, left:right, :]

def preprocess(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.py_function(crop_center, [image], tf.float32)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, (150, 150))
    return image, label

# ==========================================================
# 🧩 LOAD DATASETS
# ==========================================================
BATCH_SIZE = 32
IMG_SIZE = (150, 150)

train_dataset = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Apply cropping
train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

# ==========================================================
# 🧠 BUILD MODEL (Xception + custom layers)
# ==========================================================
base_model = Xception(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # freeze base

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ==========================================================
# 💾 CHECKPOINT SETUP (every 100 batches)
# ==========================================================
checkpoint_path = os.path.join(CHECKPOINT_DIR, "xception_batch_{batch:05d}.weights.h5")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_freq=100,  # save every 100 batches
    verbose=1
)

# Cleanup old checkpoints
def cleanup_old_checkpoints():
    checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.weights.h5")))
    if len(checkpoints) > 5:
        for ckpt in checkpoints[:-5]:
            os.remove(ckpt)

class CleanupCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            cleanup_old_checkpoints()

latest_checkpoint = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.weights.h5")))

if latest_checkpoint:
    latest_checkpoint = latest_checkpoint[-1]
    print(f"✅ Found checkpoint: {latest_checkpoint}")
    model.load_weights(latest_checkpoint)

    match = re.search(r"batch_(\d+)", latest_checkpoint)
    last_batch = int(match.group(1)) if match else 0
    print(f"🔹 Resuming from batch {last_batch}")
else:
    print("🚀 Starting training from scratch")
    last_batch = 0

EPOCHS = 20

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback, CleanupCallback()]
)

model.save(os.path.join(os.getcwd(), "final_xception_model.h5"))
print("✅ Training complete. Model saved as final_xception_model.h5")
