# =================================================================================
#
#		ml_sky_analysis - https://www.foxhollow.cc/projects/ml_sky_analysis/
#
#	 ML Sky Analysis is a tool that analyzes allsky images captured by indi-allsky
#    and estimates the sky condition (cloud coverage) using a trained keras image
#    classification model.
#
#        Copyright (c) 2024 Steve Cross <flip@foxhollow.cc>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# =================================================================================

import keras
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from keras import layers
from tensorflow import data as tf_data

logger = logging.getLogger()

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

stderr_handler = logging.StreamHandler()
stderr_handler.setFormatter(formatter)
logger.addHandler(stderr_handler)

logger.setLevel(logging.INFO)

image_size = (256, 256)
batch_size = 16
epochs = 100
seed = 503


if len(sys.argv) < 2 or (sys.argv[1] != "night" and sys.argv[1] != "day"):
    logger.error("You must specify which model, either 'night' or 'day' to train")
    exit(1)

destination_model = sys.argv[1]
train_data_dir_root = "training"
train_data_dir = os.path.join(train_data_dir_root, destination_model)
model_output_path = os.path.join(train_data_dir_root, f"{destination_model}.keras")
classes_output_path = os.path.join(train_data_dir_root, f"{destination_model}.classes")

if not os.path.isdir(train_data_dir):
    logger.error(f"Could not find the source training directory '{train_data_dir}'")
    exit(1)


# List all directories (classes) under the training folder
class_names = [d for d in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir, d))]
class_names.sort()
class_count = len(class_names)

if class_count < 1:
    logger.error(f"No class directories found in the directory '{train_data_dir}'")
    exit(1)

logger.info(f"Classes found for model '{destination_model}'")
for class_name in class_names:
    logger.info(f"  - {class_name}")


## Prepare the training dataset
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="both",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
)

# Apply `data_augmentation` to the training images.
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical"),
    layers.RandomRotation(0.1),
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)

    return images

train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)

normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

# create the model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=image_size + (3,)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(class_count, activation='softmax')  # 10 classes
])


callbacks = [
    keras.callbacks.ModelCheckpoint(os.path.join(train_data_dir_root, destination_model + "_epoch_{epoch}.keras")),
]

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

logger.info(f"Writing classes file to '{classes_output_path}'")
with open(classes_output_path, "w") as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

logger.info("Beginning training...")
## Train the model
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

# Evaluate the model
loss, accuracy = model.evaluate(val_ds)
logger.info(f"Validation Loss: {loss}")
logger.info(f"Validation Accuracy: {accuracy}")

logger.info(f"Saving model to '{model_output_path}'")
model.save()
model.summary()

logger.info(f"Writing classes file to '{classes_output_path}'")
with open(classes_output_path, "w") as f:
    f.writelines(claass_dirs)
