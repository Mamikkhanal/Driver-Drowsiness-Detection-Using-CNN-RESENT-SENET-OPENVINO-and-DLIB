import cv2
import numpy as np
import dlib
from openvino.runtime import Core

# Paths for the OpenVINO and dlib models
model_face_detection = "face-detection-adas-0001/face-detection-adas-0001.xml"
model_head_pose = "head-pose-estimation-adas-0001/head-pose-estimation-adas-0001.xml"
dlib_model = "shape_predictor_68_face_landmarks.dat"

# Initialize OpenVINO runtime
ie = Core()

# Load the models
face_detection_net = ie.read_model(model=model_face_detection)
face_detection_exec_net = ie.compile_model(model=face_detection_net, device_name="CPU")

#Load dlib model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_model)

head_pose_net = ie.read_model(model=model_head_pose)
head_pose_exec_net = ie.compile_model(model=head_pose_net, device_name="CPU")

# Get input and output layers for face detection model
face_input_layer = face_detection_exec_net.input(0)
face_output_layer = face_detection_exec_net.output(0)

# Get input and output layers for head pose estimation model
head_pose_input_layer = head_pose_exec_net.input(0)
yaw_output_layer = head_pose_exec_net.output(0)  # Yaw
pitch_output_layer = head_pose_exec_net.output(1)  # Pitch
roll_output_layer = head_pose_exec_net.output(2)  # Roll

# Function to preprocess input image
def preprocess_input(image, input_shape):
    n, c, h, w = input_shape
    resized_image = cv2.resize(image, (w, h))
    transposed_image = resized_image.transpose((2, 0, 1))  # HWC to CHW
    return np.expand_dims(transposed_image, axis=0)

# Function to draw axis on the image (for head pose visualization)

# Initialize video capture (0 for default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the input frame for face detection
    input_data = preprocess_input(frame, face_input_layer.shape)

    # Perform face detection
    face_detection_result = face_detection_exec_net([input_data])[face_output_layer]
    
    # Loop through detected faces
    for detection in face_detection_result[0][0]:
        confidence = detection[2]
        if confidence > 0.5:
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])

            # Extract face ROI
            face = frame[ymin:ymax, xmin:xmax]

            # Preprocess the face ROI for head pose estimation
            head_pose_input = preprocess_input(face, head_pose_input_layer.shape)

            # Perform head pose estimation
            yaw = head_pose_exec_net([head_pose_input])[yaw_output_layer][0][0]
            pitch = head_pose_exec_net([head_pose_input])[pitch_output_layer][0][0]
            roll = head_pose_exec_net([head_pose_input])[roll_output_layer][0][0]

            # Draw bounding box and axis
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            

    # Display the result
    cv2.imshow("Face Detection and Head Pose Estimation", frame)


    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
