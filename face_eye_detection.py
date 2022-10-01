import cv2
import sys
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
video_capture = cv2.VideoCapture(0)
while True:
    ret, img = video_capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Face = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in Face:
         cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
         roi_gray = gray[y:y+h, x:x+w]
         roi_color = img[y:y+h, x:x+w]
         eyes = eye_cascade.detectMultiScale(roi_gray)
         for (ex,ey,ew,eh) in eyes:
             cv2.rectangle(roi_color,(ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
   # Display the resulting frame
    cv2.imshow('Face and Eye Detected', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):#The 0xFF in this scenario is representing binary 11111111 a 8 bit binary,
        # since we only require 8 bits to represent a character we AND waitKey(0) to 0xFF.
        #  As a result, an integer is obtained below 255
        break
#cv2.waitKey(0)
video_capture.release()
cv2.destroyAllWindows()