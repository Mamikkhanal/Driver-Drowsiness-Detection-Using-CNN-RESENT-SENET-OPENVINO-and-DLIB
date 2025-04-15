import cv2
import numpy as np
import tensorflow as tf
from model import getModel  # Import the model function
from helperFunctions import put_text
import winsound  # Import winsound for beep sound
import threading  # Import threading to handle beeping in a separate thread
import tkinter as tk  # Import tkinter for GUI

# Load the model
model = getModel()

# Path to the weights file
weights_path = 'D:/Project III/Code/DDD/base_model/final_model_weight-app.weights.h5'

# Print the path to verify
print(f"Loading weights from: {weights_path}")

try:
    model.load_weights(weights_path)  # Change this to the actual path of your saved model weights
    print("Weights loaded successfully.")
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
    exit(1)

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 is the default camera.

drowsy = False  # Variable to track if the driver is drowsy
beeping = False  # Variable to track if the beeping is ongoing

def beep():
    while drowsy:
        winsound.Beep(1000, 1000)  # Frequency 1000 Hz, Duration 1000 ms

# Function to handle the close button click
def on_close():
    global running
    running = False
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

# Initialize the tkinter window
root = tk.Tk()
root.title("Control Panel")
close_button = tk.Button(root, text="Close", command=on_close)
close_button.pack()

# Run the tkinter window in a separate thread
running = True
tk_thread = threading.Thread(target=root.mainloop)
tk_thread.start()

while running:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    input_frame = cv2.resize(frame, (128, 128))
    input_frame = input_frame.astype('float32') / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)  # Add batch dimension
    
    # Predict using the model
    prediction = model.predict(input_frame)
    predicted_class = 'Drowsy' if prediction > 0.5 else 'Non-Drowsy'
    confidence = prediction[0][0]
    
    # Annotate the frame
    annotated_frame = put_text(np.array([frame]), [predicted_class], [confidence])[0]
    
    # Check if the driver is drowsy
    if predicted_class == 'Drowsy':
        if not drowsy:
            drowsy = True
            if not beeping:  # Start beeping if not already beeping
                beeping = True
                beep_thread = threading.Thread(target=beep)
                beep_thread.start()
    else:
        drowsy = False
        beeping = False
    
    # Display the frame
    cv2.imshow('Live Classification', annotated_frame)
    
    # Break the loop on 'b' key press
    if cv2.waitKey(1) & 0xFF == ord('b'):
        break

# Release the video capture object and close display windows
cap.release()
cv2.destroyAllWindows()
root.quit()
