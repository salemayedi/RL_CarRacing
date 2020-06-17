import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""

class CNN(nn.Module):

    def __init__(self, history_length=0, n_classes=3): 
        super(CNN, self).__init__()
        self.batchnorm = nn.BatchNorm2d(3)
        self.conv1 = nn.conv2d(1 , 6, 5) # cause the input is gray
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.conv2d(6 , 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, n_classes)

        # TODO : define layers of a convolutional neural network

    def forward(self, x):
        # TODO: compute forward pass
        x = self.batchnorm(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # check output
        return x

