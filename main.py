import cv2 as cv
import mediapipe as mp
import time
import torch as nn
import model

#Intialize variables for hand detection
mpHands = mp.solutions.hands
hands =  mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

#Selects either GPU or CPU to run on
device = nn.device("cuda" if nn.cuda.is_available() else "cpu")

#Hyperparameters
numNodes = 40000 #Size of each photo in training data is 200x200
numClasses = 28 #Number of alphabet + space + nothing
learningRate = 0.001
batchSize = 100
numEpoch = 10

#Loading data

#Takes in a image and returns a 200x200 photo of the hand with the palm as the midpoint
def handDetector(image):
    imgRGB = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    handDetect = results.multi_hand_landmarks
    
    if handDetect:
        handDetect[0]
    else:
        print("Hand not detected")

    

#Intialize network
neuroNetwork = model.network(numNodes = numNodes, numClasses = numClasses).to(device)



