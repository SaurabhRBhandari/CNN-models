from torch import nn
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.features=nn.Sequential(
            
                                 nn.Conv2d(3,64,kernel_size=3, stride=2, padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2),
                                 nn.Conv2d(64,192,kernel_size=3,padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2),
                                 
                                 nn.Conv2d(192,384,kernel_size=3,padding=1),
                                 nn.ReLU(),
                                 
                                 nn.Conv2d(384,256,kernel_size=3,padding=1),
                                 nn.ReLU(),
                                 
                                 nn.Conv2d(256,256,kernel_size=3,padding=1),
                                 nn.ReLU(),
                                 
                                 nn.MaxPool2d(kernel_size=2),
                                 nn.Flatten()
                                )
        
        self.classifier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*2*2,4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096,10),
            nn.Softmax(dim=1)
            
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x