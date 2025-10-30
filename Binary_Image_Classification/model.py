import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcreteCrackClassifier(nn.Module):
    '''
        Binary Classification Task
    '''

    def __init__(self):

        # Inherit methods of parent class
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 3)     # (BS,1, 32, 32) 
        self.pool = nn.MaxPool2d(2,2)               # (BS, 6, 15, 15)
        self.conv2 = nn.Conv2d(6, 16, 3)            # (BS, 16, 13, 13)
        self.fcl1 = nn.Linear(16*6*6, 128)                  # After pool layer (BS, 16, 6, 6)
        self.fcl2 = nn.Linear(128, 64)
        self.fcl3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):

        # First convolutional block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Second convolutional block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten layer
        x = torch.flatten(x, 1)

        # Fully connected layer
        x = self.fcl1(x)
        x = F.relu(x)

        # Fully connected layer
        x = self.fcl2(x)
        x = self.relu(x)

        # Fully connected layer
        x = self.fcl3(x)

        # Final activation function
        x = self.sigmoid(x)

        return x

