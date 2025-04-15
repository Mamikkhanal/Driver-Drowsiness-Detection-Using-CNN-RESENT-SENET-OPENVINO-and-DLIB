import os  # Import os module to handle file operations
import numpy as np
import tensorflow as tf
from model import getModel

# Replace with your actual validation data loading
# Example dummy data; replace with real validation data
validation_data = np.random.random((100, 128, 128, 3))  # 100 samples of 128x128 RGB images
validation_labels = np.random.randint(0, 2, 100)  # 100 binary labels

# Directory containing the model weights
weights_dir = 'D:/Project III/Code/DDD/base_model/'

# Initialize variables to track the best model
best_val_loss = float('inf')
best_weights_path = None

# List all weight files in the directory
weight_files = [f for f in os.listdir(weights_dir) if f.endswith('.h5')]
print(f"Found weight files: {weight_files}")

# Evaluate each model weight
for weight_file in weight_files:
    if not weight_file.startswith('model_weights.'):
        continue  # Skip non-weight files

    weights_path = os.path.join(weights_dir, weight_file)
    print(f"Evaluating weights: {weights_path}")

    # Load the model
    model = getModel()
    try:
        model.load_weights(weights_path)
    except Exception as e:
        print(f"Error loading weights: {e}")
        continue

    # Evaluate on validation data
    try:
        val_loss, val_accuracy = model.evaluate(validation_data, validation_labels, verbose=0)
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
    final_model = getModel()
    try:
        final_model.load_weights(best_weights_path)
        final_weights_path = os.path.join(weights_dir, 'final_model_weight.weights.h5')
        final_model.save_weights(final_weights_path)
        print(f"Final model weights saved to: {final_weights_path}")
    except Exception as e:
        print(f"Error loading/saving final weights: {e}")
else:
    print("No valid weight files found or all evaluations failed.")
