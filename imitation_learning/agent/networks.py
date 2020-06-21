import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""

class CNN(nn.Module):

    def __init__(self, history_length=1, n_classes=5): 
        super(CNN, self).__init__()
        self.batchnorm0 = nn.BatchNorm2d(history_length)
        self.conv1 = nn.Conv2d(history_length , 6, 7) # batch * 6 * 90 *90
        self.pool1 = nn.MaxPool2d(5, 5) # batch * 6 * 18 *18    
        self.batchnorm1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6 , 16, 3) # batch * 16 * 16 * 16
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2) # batch * 16 * 8 * 8 # or 4 and becomes 16*4*4
        self.fc1 = nn.Linear(16 * 8 * 8, 128)
        self.drouput = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, n_classes)

        # TODO : define layers of a convolutional neural network

    def forward(self, x):
        # TODO: compute forward pass
        x = self.batchnorm0(x)
        x = self.batchnorm1(self.conv1(x))
        x = self.pool1(F.relu(x)) # batch before relu !
        x = self.batchnorm2(self.conv2(x))
        x = self.pool2(F.relu(x))
        x = x.contiguous()
        x = x.view(-1, 16 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.drouput(x)
        x = self.fc2(x) # check output
        return x

