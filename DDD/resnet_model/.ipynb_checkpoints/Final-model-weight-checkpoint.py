import os  # Import os module to handle file operations
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Define function to create the model
def create_model():
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False, weights='imagenet', input_shape=(144,144,3)
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

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

# Directory containing the model weights
weights_dir = 'D:/Project III/Code/DDD/resnet_model/Model_Weights'

# Initialize variables to track the best model
best_val_loss = float('inf')
best_weights_path = None

# List all weight files in the directory
weight_files = [f for f in os.listdir(weights_dir) if f.endswith('.weights.h5')]
print(f"Found weight files: {weight_files}")

# Evaluate each model weight
for weight_file in weight_files:
    if not weight_file.startswith('model.'):
        continue  # Skip non-weight files

    weights_path = os.path.join(weights_dir, weight_file)
    print(f"Evaluating weights: {weights_path}")

    # Load the model
    model = create_model()
    try:
        model.load_weights(weights_path)
    except Exception as e:
        print(f"Error loading weights: {e}")
        continue

    # Evaluate on validation data
    try:
        val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)
        print(f"Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")

        # Track the best model weight
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights_path = weights_path
    except Exception as e:
        print(f"Error evaluating model: {e}")

# Print and save the best model weight
if best_weights_path:
    print(f"Best weights found: {best_weights_path} with validation loss: {best_val_loss}")
    
    # Load and save the best model weight as the final model weight
    final_model = create_model()
    try:
        final_model.load_weights(best_weights_path)
        final_weights_path = os.path.join(weights_dir, 'final_model_weight.weights.h5')
        final_model.save_weights(final_weights_path)
        print(f"Final model weights saved to: {final_weights_path}")
    except Exception as e:
        print(f"Error loading/saving final weights: {e}")
else:
    print("No valid weight files found or all evaluations failed.")
