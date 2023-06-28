import cv2 as cv
import mediapipe as mp
import time
import torch as nn

#make a video capture object that connects to the camera
cap = cv.VideoCapture(0)
#gets hand data
mpHands = mp.solutions.hands
hands =  mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

#Creates the video
while True:
     #reads the camera
     success, img = cap.read()
     #converts from BGR to RGB
     imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
     #process() uses RGB only
     results = hands.process(imgRGB)
     
     #Checks if there are hands
     handDetect = results.multi_hand_landmarks
     if handDetect:
          #May be multiple hands
          for handLms in handDetect:
               for id,lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    centerX, centerY  = int(lm.x*w),int(lm.y*h)
                    
               mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


     cTime = time.time()
     fps = 1/(cTime-pTime)
     pTime = cTime
     
     cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)
     
     cv.imshow("video",img)
     cv.waitKey(1)