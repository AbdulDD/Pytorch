import torch.nn as nn

LATENT_DIMENSION = 128

# Encoder
class Encoder(nn.Module):
    def __init__(self):
        '''
        init function
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) # output dimension (BS, 6, 62, 62)
        self.conv2 = nn.Conv2d(6, 16, 3) # (BS, 16, 60, 60)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(16*60*60, LATENT_DIMENSION)
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        
        return x

# Decoder
class  Decoder(nn.Module):
    def __init__(self):
        '''
        init function
        '''
        super().__init__()
        self.linear = nn.Linear(LATENT_DIMENSION, 16*60*60)
        self.deconv2 = nn.ConvTranspose2d(16, 6, 3)
        self.deconv1 = nn.ConvTranspose2d(6, 1, 3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 16, 60, 60) # (BS, Features) to (BS, Channels, width, height)
        x = self.deconv2(x)
        x = self.relu(x)
        x = self.deconv1(x)
        x = self.relu(x)

        return x

# AutoEncoder
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x




