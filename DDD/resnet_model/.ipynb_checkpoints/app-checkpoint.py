# Import packages and set numpy random seed
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

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
    target_size=(144, 144),
    batch_size=64,
    shuffle=True,
    subset='training',
    class_mode='binary')

print(train_generator)

validation_generator = train_datagen.flow_from_directory(
    '../Datasets/train_data/',
    target_size=(144, 144),
    batch_size=64,
    shuffle=True,
    subset='validation',
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    '../Datasets/train_data/',
    target_size=(144, 144),
    batch_size=64,
    shuffle=True,
    class_mode='binary')

# Load the ResNet50V2 model
base_model = tf.keras.applications.ResNet50V2(
    include_top=False, weights='imagenet', input_shape=(144,144,3)
)

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer and a logistic layer for binary classification
x = Dense(1, activation='sigmoid')(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Summary of the model
model.summary()

# Logging and callbacks
my_callbacks = [
    # tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.weights.h5', 
                                       save_weights_only=True,
                                       monitor='val_accuracy'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs', update_freq='batch', profile_batch=0),
]

# Train the model
hist = model.fit(train_generator,
                 epochs=20,
                 validation_data=validation_generator,
                 validation_steps=validation_generator.samples // validation_generator.batch_size,
                 callbacks=my_callbacks)

# Test the model
score = model.evaluate(test_generator, verbose=1)
print('Test accuracy:', score[1])

# Plot training & validation accuracy values
import matplotlib.pyplot as plt

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
