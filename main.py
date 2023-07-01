import cv2 as cv
import mediapipe as mp
import time
import torch as nn
import model

#Selects either GPU or CPU to run on
device = nn.device("cuda" if nn.cuda.is_available() else "cpu")

#Hyperparameters
numNodes = 40000 #Size of each photo in training data is 200x200
numClasses = 28 #Number of alphabet + space + nothing
learningRate = 0.001
batchSize = 100
numEpoch = 10

#Loading data


#Intialize network
neuroNetwork = model.network(numNodes = numNodes, numClasses = numClasses).to(device)


