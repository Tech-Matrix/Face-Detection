import cv2
import numpy as np



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

