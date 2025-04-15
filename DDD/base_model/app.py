import model  # from model.py
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras

import matplotlib.pyplot as plt

# load data
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

print(train_generator)

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


# get model from model.py
baseModel = model.getModel()


# Train the model
my_callbacks = [
    # tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model_weights.{epoch:02d}-{val_loss:.2f}.weights.h5',
                                       save_weights_only=True,
                                       monitor='val_accuracy'),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', update_freq='batch', profile_batch=0)
]

# The model weights (that are considered the best) are loaded into the model.
# model.load_weights(checkpoint_filepath)


# Train the model
hist = baseModel.fit(train_generator,
                     epochs=20,
                     batch_size=train_generator.samples,
                     validation_data=validation_generator,
                     validation_steps=validation_generator.samples,
                     verbose=1,
                     callbacks=my_callbacks)

# test
score = baseModel.evaluate(test_generator,verbose=1)
print('Test accuracy:', score[1])

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(hist.hist['accuracy'])
plt.plot(hist.hist['val_accuracy'])
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
