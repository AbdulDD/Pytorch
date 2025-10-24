import torch.nn as nn
import torch

class IrisClassification(nn.Module):
    def __init__(self, num_features, num_hiddenstate, num_classes):
        super(IrisClassification, self).__init__()
        
        self.linear1 = nn.Linear(num_features, num_hiddenstate)
        self.linear2 = nn.Linear(num_hiddenstate, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
 
    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
