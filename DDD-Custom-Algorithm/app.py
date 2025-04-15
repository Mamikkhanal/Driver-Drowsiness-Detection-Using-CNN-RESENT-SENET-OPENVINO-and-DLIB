import cv2
from openvino.runtime import Core
from imutils import face_utils
import dlib
import numpy as np
from datetime import datetime, timedelta
import pygame

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

# global color variable for the color of face bounding box
color = (0, 255, 0)

# Function to preprocess input image
def preprocess_input(image, input_shape):
    n, c, h, w = input_shape
    resized_image = cv2.resize(image, (w, h))
    transposed_image = resized_image.transpose((2, 0, 1))  # HWC to CHW
    return np.expand_dims(transposed_image, axis=0)	

def hisEqulColor(img):
	hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
	channels=cv2.split(hls)
	# print(np.mean(channels[0]))
	if(np.mean(channels[1] ) < 127):
		# clahe = cv2.createCLAHE(clipLimit=16.0,tileGridSize=(8,8))
		# channels[1] = clahe.apply(channels[1])
		cv2.equalizeHist(channels[1],channels[1])
		cv2.merge(channels,hls)
		cv2.cvtColor(hls,cv2.COLOR_HLS2BGR,img)
		# print("after equ "+str(np.mean(cv2.split(yuv)[0])))
	
	return img

def get_facial_points(frame, face_detection_result):
	confidence = face_detection_result[2]
	if confidence > 0.5:
		xmin = int(face_detection_result[3] * frame.shape[1])
		ymin = int(face_detection_result[4] * frame.shape[0])
		xmax = int(face_detection_result[5] * frame.shape[1])
		ymax = int(face_detection_result[6] * frame.shape[0])
		return xmin, xmax, ymin, ymax
	else:
		return None

def apply_CLAHE(image_of_face):
	crop_dlib = cv2.resize(image_of_face, (300,300))
	gray = cv2.cvtColor(crop_dlib, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
	gray = clahe.apply(gray)
	return gray

def get_facial_coordinates(face):
	rects = detector(face,0)
	return rects

def main():
	
	total_count = 0

	#for sound
	pygame.mixer.init()
	pygame.mixer.set_num_channels(8)
	voice = pygame.mixer.Channel(5)
	sound = pygame.mixer.Sound("beep.mp3")

	#eye counters
	time = datetime.now()
	one_min_start_time = time
	eye_closed_counter = 0
	eye_closed = False
	eye_close_time = time

	#mouth counters
	mouth_open_time = time
	mouth_open = False
	yawn_count = 0

	#nod counters
	nod_time = time
	nodding = False
	nod_count = 0

	m_EAR_left, m_EAR_right = get_mEARS()
	cap = cv2.VideoCapture(0)
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		# for mirror image
		frame = cv2.flip(frame, 1)
		# histogram equalization for mean brightness
		frame = hisEqulColor(frame)
		
		# Preprocess the input frame for face detection
		input_data = preprocess_input(frame, face_input_layer.shape)

		# Perform face detection
		face_detection_result = face_detection_exec_net([input_data])[face_output_layer]
		
		# taking the first detected face
		# the first face always has highest confidence
		detected_face = face_detection_result[0][0][0]
		facial_points = get_facial_points(frame, detected_face)
		if facial_points:
			# Extract face ROI
			xmin, xmax, ymin, ymax = facial_points
			# taking 30 pixels beyound the face
			face = frame[ymin - 30 : ymax + 30, xmin - 30 : xmax + 30]
			#Drawing the box with image
			cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),color,2)
		else:
			# don't do any other thing unless a face is found
			print("No face detected")
			continue

		# Preprocess the face ROI for head pose estimation
		head_pose_input = preprocess_input(face, head_pose_input_layer.shape)

		# Perform head pose estimation
		# we need pitch only, so ignoring others
		pitch = head_pose_exec_net([head_pose_input])[pitch_output_layer][0][0]
		# yaw = head_pose_exec_net([head_pose_input])[yaw_output_layer][0][0]
		# roll = head_pose_exec_net([head_pose_input])[roll_output_layer][0][0]

		if pitch > -15 and pitch < 25:
			cv2.putText(frame,"Straight face", (20,60),cv2.FONT_HERSHEY_COMPLEX, 1, color,1)
			nodding = False

		elif pitch > 25:
			cv2.putText(frame,"Facing downwards", (20,60),cv2.FONT_HERSHEY_COMPLEX, 1, color,1)
			if not nodding:
				nodding = True
				nod_time = time.now()

			if nodding:
				if datetime.now() - nod_time >= timedelta(seconds = 2):
					nod_count += 1
					print("nod count= "+ str(nod_count))
					nod_time = time.now()
					if voice.get_busy() == 0:
						voice.play(sound)

		elif pitch < -15:
			cv2.putText(frame,"Facing upwards", (20,60),cv2.FONT_HERSHEY_COMPLEX, 1, color,1)
			nodding = False
		
	 
		#feed frame face to dlib
		gray = apply_CLAHE(face)
		rects = get_facial_coordinates(gray)
		for (i, rect) in enumerate(rects):
			xmin = rect.left()
			ymin = rect.top()
			xmax = rect.right()
			ymax = rect.bottom()

			cv2.rectangle(gray,(xmin,ymin),(xmax,ymax),color,1)
			coordinates = predictor(gray, rect)
			coordinates = face_utils.shape_to_np(coordinates)

			#EARs
			ear_l = ((coordinates[41][1]-coordinates[37][1]) + (coordinates[40][1]-coordinates[38][1]))/(2*(coordinates[39][0]-coordinates[36][0]))
			ear_r = ((coordinates[47][1]-coordinates[43][1]) + (coordinates[46][1]-coordinates[44][1]))/(2*(coordinates[45][0]-coordinates[42][0]))

			if(ear_l < m_EAR_left and ear_r < m_EAR_right):
				cv2.putText(frame, "Eyes closed", (20,30),cv2.FONT_HERSHEY_COMPLEX, 1, color, 1)

				#if eye closed for first frame set the eye close time(eye was not closed in last frame)
				if not eye_closed:
					eye_close_time = datetime.now()

				#if eye closed for more than 2 sec straight
				if eye_closed and (datetime.now() - eye_close_time >= timedelta(seconds = 2)):
					if voice.get_busy() == 0:
						voice.play(sound) 
					
				eye_closed = True

				
			else:
				cv2.putText(frame, "Eyes open", (20,30),cv2.FONT_HERSHEY_COMPLEX, 1,color, 1)

				#previous frame was eye closed and now eye opened, increase counter
				if eye_closed:
					eye_closed_counter += 1
					print("eye_closed_counter = "+ str(eye_closed_counter))
				eye_closed = False


			#mouth ratio

			mouth_ratio = ((coordinates[58][1] - coordinates[50][1]) + (coordinates[56][1] - coordinates[52][1])) / (2*(coordinates[54][0] - coordinates[48][0]))

			for (x,y) in coordinates:
				cv2.circle(gray, (x,y), 2, color, -1)

			if(mouth_ratio>0.35):
				cv2.putText(frame,"Mouth open", (20,90),cv2.FONT_HERSHEY_COMPLEX, 1, color, 1)
				#if previous frame was closed, then first frame to open mouth note time
				if not mouth_open:
					mouth_open_time = datetime.now()
					mouth_open = True
					#print(mouth_open_time)
				#check if more than 4.5 sec opened
				if mouth_open and (datetime.now() - mouth_open_time > timedelta(seconds = 4.5)):
					print(datetime.now() - mouth_open_time)
					mouth_open_time = datetime.now()
					yawn_count += 1
					print("yawn count= "+ str(yawn_count))
					if voice.get_busy() == 0:
						voice.play(sound)

			else:
				cv2.putText(frame,"Mouth close", (20,90),cv2.FONT_HERSHEY_COMPLEX, 1, color, 1)
				if mouth_open:
					mouth_open = False
					voice.stop()

		#check 1 min timer:
		if(datetime.now() - one_min_start_time >= timedelta(minutes = 1)):
			one_min_start_time = datetime.now()
			if eye_closed_counter > 25 or eye_closed_counter < 10:
				total_count += 1
				if(voice.get_busy() == 0):
					voice.play(sound)


			total_count += yawn_count + nod_count
			print("drowsy_count = "+ str(total_count))
			print("total eye_closed_counter= "+ str(eye_closed_counter))
			print("total yawn = " + str(yawn_count))
			print("total nod_count = " + str(nod_count))

			if total_count >= 3:
				voice.play(sound)

			print(eye_closed_counter)
			yawn_count = 0
			eye_closed_counter = 0
			nod_count = 0

		cv2.imshow("Face detection", frame)
		cv2.imshow("Cropped face", gray)
		
		#to stop
		interrupt = cv2.waitKey(10)
		if interrupt & 0xFF == 27:
			break
	cap.release()
	cv2.destroyAllWindows()


def get_mEARS():
	cap = cv2.VideoCapture(0)
	start_time = datetime.now()
	total_frames_with_face = 0

	l_Amin = l_Bmin = l_Cmin = r_Amin = r_Bmin = r_Cmin = 100
	l_Amax = l_Bmax = l_Cmax = r_Amax = r_Bmax = r_Cmax = 0

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		#for mirror image
		frame = cv2.flip(frame, 1)
		frame = hisEqulColor(frame)
		
		# Preprocess the input frame for face detection
		input_data = preprocess_input(frame, face_input_layer.shape)

		# Perform face detection
		face_detection_result = face_detection_exec_net([input_data])[face_output_layer]

		# taking the first detected face
		# the first face always has highest confidence
		detected_face = face_detection_result[0][0][0]
		facial_points = get_facial_points(frame, detected_face)
		if facial_points:
			# Extract face ROI
			xmin, xmax, ymin, ymax = facial_points
			# taking 30 pixels beyound the face
			face = frame[ymin - 30 : ymax + 30, xmin - 30 : xmax + 30]
			total_frames_with_face += 1
			#Drawing the box with image
			cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),color,1)
		else:
			# don't do any other thing unless a face is found
			print("No face detected")
			continue

		#feed frame face to dlib
		gray = apply_CLAHE(face)
		rects = get_facial_coordinates(gray)
		for (i, rect) in enumerate(rects):
			xmin = rect.left()
			ymin = rect.top()
			xmax = rect.right()
			ymax = rect.bottom()

			cv2.rectangle(gray,(xmin,ymin),(xmax,ymax),color,1)
			coordinates = predictor(gray, rect)
			coordinates = face_utils.shape_to_np(coordinates)

			#EAR			
			left_A = coordinates[41][1]-coordinates[37][1]
			left_B = coordinates[40][1]-coordinates[38][1]			
			left_C = coordinates[39][0]-coordinates[36][0]

			right_A = coordinates[47][1]-coordinates[43][1]		
			right_B = coordinates[46][1]-coordinates[44][1]			
			right_C = coordinates[45][0]-coordinates[42][0]

			#left
			l_Amin = min(left_A, l_Amin)
			l_Amax = max(left_A, l_Amax)

			l_Bmin = min(left_B, l_Bmin)
			l_Bmax = max(left_B, l_Bmax)

			l_Cmin = min(left_C, l_Cmin)
			l_Cmax = max(left_C, l_Cmax)

			#right
			r_Amin = min(right_A, r_Amin)
			r_Amax = max(right_A, r_Amax)

			r_Bmin = min(right_B, r_Bmin)
			r_Bmax = max(right_B, r_Bmax)

			r_Cmin = min(right_C, r_Cmin)
			r_Cmax = max(right_C, r_Cmax)

			cv2.putText(frame, "l_Amin\tl_Amax\tl_Bmin\tl_Bmax\tl_Cmin\tl_Cmax", (100,50),cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))
			cv2.putText(frame, str(l_Amin)+"\t"+str(l_Amax)+"\t"+str(l_Bmin)+"\t"+str(l_Bmax)+"\t"
				+str(l_Cmin)+"\t"+str(l_Cmax), (100,65),cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))

			for (x,y) in coordinates:
				cv2.circle(gray, (x,y), 2, color, -1)


		cv2.imshow("Face detection", frame)
		cv2.imshow("Cropped face", gray)

		# calculate meadian EARs
		ear_close_left = (l_Amin + l_Bmin) / (2 * l_Cmax)
		ear_open_left = (l_Amax + l_Bmax) / (2 * l_Cmin)
		m_EAR_left = (ear_open_left - ear_close_left) / 2

		ear_close_right = (r_Amin + r_Bmin) / (2 * r_Cmax)
		ear_open_right = (r_Amax + r_Bmax) / (2 * r_Cmin)
		m_EAR_right = (ear_open_right - ear_close_right) / 2
		
		#to stop
		interrupt = cv2.waitKey(10)
		if interrupt & 0xFF == 27:
			break
		if(datetime.now() - start_time > timedelta(seconds=10) and total_frames_with_face > 100):
			break
	cap.release()
	cv2.destroyAllWindows()
	print(m_EAR_left)
	print(m_EAR_right)
	return m_EAR_left, m_EAR_right


if __name__ == '__main__':
	main()