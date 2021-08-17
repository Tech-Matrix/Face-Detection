
import cv2

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,1050)
cap.set(10,100)
while True:
    success, img = cap.read()

    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces=face_cascade.detectMultiScale(gray, 1.8, 5)

    for (x,y,w,h) in faces:
         cv2.rectangle(img, (x, y), (x+w, y+h), (252, 1, 2), 3)

    cv2.imshow("video",img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
     break
cap.release()

