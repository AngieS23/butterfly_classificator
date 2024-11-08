import torch.nn as nn
import torch.nn.functional as functional

class ConvNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super(ConvNet, self).__init__()

        # Layers
        self.layer_1 = nn.Conv2d(in_channels=in_channels, 
                                 out_channels=8, 
                                 kernel_size=3, 
                                 stride=1, 
                                 padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, 
                                 stride=2)
        self.layer_2 = nn.Conv2d(in_channels=8, 
                                 out_channels=16, 
                                 kernel_size=3, 
                                 stride=1, 
                                 padding=1)
        self.fully_connected = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = functional.relu(self.layer_1(x)) 
        x = self.pool(x)
        x = functional.relu(self.layer_2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fully_connected(x)
        return x