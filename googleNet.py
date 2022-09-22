from torch import nn
import torch

class InceptionBlock(nn.Module):
    def __init__(self,in_channels,c1x1,c3x3,c5x5,c3x3r,c5x5r,pool_proj):
        super(InceptionBlock,self).__init__()
        self.p1=nn.Sequential(
            nn.Conv2d(kernel_size=1,in_channels=in_channels,out_channels=c1x1),
            nn.ReLU(),
            )
        self.p2=nn.Sequential(
            nn.Conv2d(in_channels,out_channels=c3x3r,kernel_size=1),
            nn.Conv2d(in_channels=c3x3r,out_channels=c3x3,kernel_size=3,padding=1),
            nn.ReLU(),
        )
        self.p3=nn.Sequential(
            nn.Conv2d(in_channels,out_channels=c5x5r,kernel_size=1),
            nn.Conv2d(in_channels=c5x5r,out_channels=c5x5,kernel_size=5,padding=2),
            nn.ReLU(),
        )
        self.p4=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=in_channels,kernel_size=1,out_channels=pool_proj),
            nn.ReLU(),
        )
    def forward(self,x):
        return torch.cat([self.p1(x),self.p2(x),self.p3(x),self.p4(x)],1)       

class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet,self).__init__()
        self.stack =nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(kernel_size=3,in_channels=64,out_channels=192,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5,stride=1),
            InceptionBlock(in_channels=192,c1x1=64,c3x3r=96,c3x3=128,c5x5r=16,c5x5=32,pool_proj=32),
            InceptionBlock(256,128,192,96,128,32,64),
            nn.MaxPool2d(kernel_size=3,stride=2),
            InceptionBlock(480,192,208,48,96,16,64),
            InceptionBlock(512,160,224,64,112,24,64),
            InceptionBlock(512,128,256,64,128,24,64),
            InceptionBlock(512,112,288,64,144,32,64),
            InceptionBlock(528,256,320,128,160,32,128),
            nn.MaxPool2d(kernel_size=3,stride=2),
            InceptionBlock(832,256,320,128,160,32,128),
            InceptionBlock(832,384,384,128,192,48,128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=6),
            nn.Flatten(),
            nn.Linear(in_features=1024,out_features=10),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        
        return self.stack(x)