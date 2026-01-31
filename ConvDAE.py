import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def conv_block(cin,cout,dropout = 0.0):
    layers = [nn.Conv2d(cin,cout,kernel_size = 3, padding = 1),
              nn.ReLU(inplace = True),
              nn.Conv2d(cout,cout,kernel_size = 3, padding = 1),
              nn.ReLU(inplace = True)]
    if dropout and dropout>0:
        layers.insert(2,nn.Dropout2d(dropout))
    
    return nn.Sequential(*layers)  # '*' to give each item in the list as arg and not the whole list

class Autoencoder(nn.Module):
    def __init__(self, dropout = 0.1, out_activation = None):
        super().__init__()
        self.activation = out_activation
        
        ## ENCODER

        # output : 64x64x16
        self.enc0 = nn.Sequential(nn.Conv2d(3,16, kernel_size=3, padding = 1),
                             nn.ReLU(inplace = True),)
        
        # output : 32x32x32
        self.down1 = nn.Conv2d(16,32, kernel_size=3, stride = 2,padding = 1)
        self.enc1 = conv_block(32,32,dropout)                               
        # output : 16x16x64
        self.down2 = nn.Conv2d(32,64, kernel_size=3, stride = 2, padding = 1)
        self.enc2 = conv_block(64,64,dropout)
        # output : 8x8x128
        self.down3= nn.Conv2d(64,128, kernel_size=3, stride = 2, padding = 1)
        self.enc3= conv_block(128,128,dropout)
        # output : 4x4x256
        self.down4 = nn.Conv2d(128,256, kernel_size=3, stride = 2, padding = 1)
        self.bottleneck = conv_block(256,256,dropout)

        #DECODER

        # output : 8x8x128
        self.up4 = nn.ConvTranspose2d(256,128,kernel_size = 4, stride = 2, padding = 1)
        self.dec3 = conv_block(128,128, dropout)
        # output : 16x16x64
        self.up3 = nn.ConvTranspose2d(128,64,kernel_size = 4, stride = 2, padding = 1)
        self.dec2 = conv_block(64,64, dropout)
        # output : 32x32x32
        self.up2 = nn.ConvTranspose2d(64,32,kernel_size = 4, stride = 2, padding = 1)
        self.dec1 = conv_block(32,32, dropout)
        # output : 64x64x16
        self.up1 = nn.ConvTranspose2d(32,16,kernel_size = 4, stride = 2, padding = 1)
        self.dec0 = nn.Sequential(nn.Conv2d(16,16,kernel_size = 3, padding = 1),
                    nn.ReLU(inplace = True))
        # output : 64x64x3
        self.out = nn.Conv2d(16,3,kernel_size = 3, padding = 1)

    def forward(self,x):
        x = self.enc0(x)                 # (B,16,64,64)
        x = F.relu(self.down1(x))        # (B,32,32,32)
        x = self.enc1(x)                 # (B,32,32,32)

        x = F.relu(self.down2(x))        # (B,64,16,16)
        x = self.enc2(x)                 # (B,64,16,16)

        x = F.relu(self.down3(x))        # (B,128,8,8)
        x = self.enc3(x)                 # (B,128,8,8)

        x = F.relu(self.down4(x))        # (B,256,4,4)
        x = self.bottleneck(x)           # (B,256,4,4)

        # Decoder
        x = F.relu(self.up4(x))          # (B,128,8,8)
        x = self.dec3(x)                 # (B,128,8,8)

        x = F.relu(self.up3(x))          # (B,64,16,16)
        x = self.dec2(x)                 # (B,64,16,16)

        x = F.relu(self.up2(x))          # (B,32,32,32)
        x = self.dec1(x)                 # (B,32,32,32)

        x = F.relu(self.up1(x))          # (B,16,64,64)
        x = self.dec0(x)                 # (B,16,64,64)

        x = self.out(x)                  # (B,3,64,64)

        if self.activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.activation == "tanh":
            x = torch.tanh(x)

        return x
    
    

        
