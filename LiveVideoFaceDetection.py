import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="display")
mode = ap.parse_args().mode

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# emotions will be displayed on your face from the webcam feed
if mode == "display":
	model.load_weights('model.h5')

	# prevents openCL usage and unnecessary logging messages
	cv2.ocl.setUseOpenCL(False)

	# dictionary which assigns each label an emotion (alphabetical order)
	emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

	# start the webcam feed
	face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	eyes_detect = cv2.CascadeClassifier('haarcascade_eye.xml')
	noise_detect = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

	capture = cv2.VideoCapture(0)


	fourcc=cv2.VideoWriter_fourcc(*'XVID')
	out=cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))


	while True:
		
		ret, capturing = capture.read()

		
		gray = cv2.cvtColor(capturing, cv2.COLOR_BGR2GRAY)
		face_detection = face_detect.detectMultiScale(gray, 1.3, 5)
		
		for (x, y, w, h) in face_detection:
			cv2.rectangle(capturing, (x, y), (x + w, y + h), (200,200, 255), 2)

			gray_roi = gray[y:y + h, x:x + w]
			color_roi = capturing[y:y + h, x:x + w]

		
			eye_detector = eyes_detect.detectMultiScale(gray_roi)

			# cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

			cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray_roi, (48, 48)), -1), 0)
			prediction = model.predict(cropped_img)
			maxindex = int(np.argmax(prediction))
			cv2.putText(capturing, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


		
			for (eye_x, eye_y, eye_w, eye_h) in eye_detector:
				cv2.rectangle(color_roi, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (255, 0, 0), 5)

			
				nose_detector = noise_detect.detectMultiScale(gray_roi, 1.3, 5)

			for (nose_x, nose_y, nose_w, nose_h) in nose_detector:
				cv2.rectangle(color_roi, (nose_x, nose_y), (nose_x + nose_w, nose_y + nose_h), (0, 255, 0), 5)

				
		if ret==True:
			out.write(capturing)

		
		cv2.imshow("Real-time Detection", capturing)

		c = cv2.waitKey(1)
		if c == 27:
			break


	capture.release()

	cv2.destroyAllWindows()
