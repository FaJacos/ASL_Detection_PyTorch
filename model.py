import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class network(nn.Module):
    def __init__(self, numNodes, numClasses):
        super(network,self).__init__()
        
        self.linear1 = nn.Linear(numNodes, 1000)
        self.linear2 = nn.Linear(1000, 250)
        self.linear3 = nn.Linear(250, numClasses)

        
    def forward(self,x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        return x