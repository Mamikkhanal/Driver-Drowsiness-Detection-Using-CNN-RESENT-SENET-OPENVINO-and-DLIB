import model  # from model.py
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.4,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '../Datasets/train_data/',
    target_size=(128, 128),
    batch_size=32,
    shuffle=True,
    subset='training',
    class_mode='binary')

validation_generator = train_datagen.flow_from_directory(
    '../Datasets/train_data/',
    target_size=(128, 128),
    batch_size=32,
    shuffle=True,
    subset='validation',
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    '../Datasets/train_data/',
    target_size=(128, 128),
    batch_size=32,
    shuffle=True,
    class_mode='binary')

# Get model from model.py
baseModel = model.getModel()

# Define callbacks (excluding ModelCheckpoint to avoid saving weights per epoch)
my_callbacks = [
    # Optional: Early stopping if needed
    # tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.TensorBoard(log_dir='./logs', update_freq='batch', profile_batch=0)
]

# Train the model
hist = baseModel.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    verbose=1,
    callbacks=my_callbacks
)

# Save the final model weights
final_weights_path = 'D:/Project III/Code/DDD/base_model/final_model_weight-app.weights.h5'
baseModel.save_weights(final_weights_path)
print(f"Final model weights saved to: {final_weights_path}")

# Test
score = baseModel.evaluate(test_generator, verbose=1)
print('Test accuracy:', score[1])

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
