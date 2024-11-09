import torch.nn as nn
import torch.nn.functional as functional

class ConvNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super(ConvNet, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten() 
        self.fc_1 = nn.Linear(64 * 49 * 49, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc_2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = functional.relu(self.conv_1(x))
        x = self.pool(x)
        
        x = functional.relu(self.conv_2(x))
        x = self.pool(x)
        
        x = self.flatten(x)
        x = functional.relu(self.fc_1(x))
        x = self.dropout(x)
        x = functional.softmax(self.fc_2(x), dim=1)
        
        return x