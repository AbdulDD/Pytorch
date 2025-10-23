import torch.nn as nn

class MPGPredictor(nn.Module):
    '''
    1. we inherit nn module functions,methods and properties using nn.module
    2. super is used to initialize the class with nn module
    '''
    def __init__(self, input_size, output_size):
        super(MPGPredictor, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)
