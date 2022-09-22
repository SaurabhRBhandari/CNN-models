from torch import nn
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.stack=nn.Sequential(nn.Conv2d(3,10,kernel_size=5),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2),
                                 nn.Conv2d(10,16,kernel_size=5),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2),
                                 nn.Flatten(),
                                 nn.Linear(16*5*5,120),
                                 nn.ReLU(),
                                 nn.Linear(120,84),
                                 nn.ReLU(),
                                 nn.Linear(84,10),
                                 nn.Softmax())

    def forward(self, x):

        return self.stack(x)