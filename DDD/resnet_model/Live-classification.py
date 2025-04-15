import cv2
import numpy as np
import tensorflow as tf
import winsound
import threading

# Define a function to load your model
def getModel():
    # Load the ResNet50V2 model
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False, weights=None, input_shape=(144, 144, 3))
    
    # Add a global spatial average pooling layer
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Add a fully-connected layer and a logistic layer for binary classification
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create the model
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    
    return model

# Load the model
model = getModel()

# Path to the weights file
weights_path = 'D:/Project III/Code/DDD/resnet_model/Model_Weights/final_model_weight.weights.h5'

# Print the path to verify
print(f"Loading weights from: {weights_path}")

try:
    model.load_weights(weights_path)
    print("Weights loaded successfully.")
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
    exit(1)
except ValueError as e:
    print(f"ValueError: {e}")
    exit(1)

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 is the default camera.

drowsy = False  # Variable to track if the driver is drowsy
beeping = False  # Variable to track if the beeping is ongoing

def beep():
    while drowsy:
        winsound.Beep(1000, 1000)  # Frequency 1000 Hz, Duration 1000 ms

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    input_frame = cv2.resize(frame, (144, 144))
    input_frame = input_frame.astype('float32') / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)  # Add batch dimension
    
    # Predict using the model
    prediction = model.predict(input_frame)
    predicted_class = 'Drowsy' if prediction > 0.5 else 'Alert'
    confidence = prediction[0][0]
    
    # Annotate the frame
    text = f"{predicted_class}: {confidence:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
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
        if beeping:
            beeping = False
    
    # Display the frame
    cv2.imshow('Live Classification', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close display windows
cap.release()
cv2.destroyAllWindows()
