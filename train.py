import cv2
import logging
import numpy
import os
import tensorflow as tf
import time

from PIL import Image
from tensorflow import keras
from tensorflow.keras import Sequential, models, layers
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, MaxPooling2D

logger = logging.getLogger()

image_size = (180, 180)
batch_size = 16
epochs = 50
seed = 123

train_data_dir = "training"

# List all directories (classes) under the training folder
class_dirs = [d for d in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir, d))]

# Count the number of directories
class_count = len(class_dirs)

print(f"Number of classes found under {train_data_dir}: {class_count}")
logger.info(f"Number of classes: {class_count}")


# Load the data
train_dataset, validation_dataset = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# Normalize the data
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_train_ds = train_dataset.map(lambda x, y: (normalization_layer(x), y))
normalized_val_ds = validation_dataset.map(lambda x, y: (normalization_layer(x), y))


# # Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(180, activation='relu'),
    Dense(class_count, activation='softmax')  # 10 classes
])

# Custom callback to log progress
class LoggingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()
        logger.info("Training started")

    def on_train_end(self, logs=None):
        train_duration = time.time() - self.train_start_time
        logger.info(f"Training finished in {train_duration:.2f} seconds")

    def on_epoch_begin(self, epoch, logs=None):
        logger.info(f"Starting epoch {epoch + 1}")

    def on_epoch_end(self, epoch, logs=None):
        logger.info(f"Finished epoch {epoch + 1}, loss: {logs['loss']}, accuracy: {logs['accuracy']}")

    def on_batch_end(self, batch, logs=None):
        logger.info(f"Finished batch {batch + 1}, loss: {logs['loss']}, accuracy: {logs['accuracy']}")


# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Model checkpoint callback
checkpoint_filepath = "best.weights.keras"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath=checkpoint_filepath,
                        save_weights_only=False,
                        monitor='val_accuracy',
                        mode='max',
                        save_best_only=True)

# Early stopping callback
# EarlyStopCallback = tf.keras.callbacks.EarlyStopping(
#                         monitor='accuracy',
#                         min_delta=0,
#                         patience=5,
#                         verbose=1,
#                         restore_best_weights=True
#                         )

# Load the model if it exists
if os.path.exists(checkpoint_filepath):
    model = tf.keras.models.load_model(checkpoint_filepath)
else:
    logger.warning(f"Checkpoint file {checkpoint_filepath} not found. Skipping model loading.")

# Callbacks
callbacks_list = [
    # EarlyStopCallback, 
                  model_checkpoint_callback, 
                  LoggingCallback()]

# Train the model
history = model.fit(normalized_train_ds, 
                    validation_data=normalized_val_ds, 
                    epochs=epochs, 
                    callbacks=callbacks_list)

# Evaluate the model
loss, accuracy = model.evaluate(normalized_val_ds)
logger.info(f"Validation Loss: {loss}")
logger.info(f"Validation Accuracy: {accuracy}")

# Save the model
model.save('saved.keras')
model.summary()