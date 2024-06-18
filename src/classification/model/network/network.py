import torch.nn as nn
from torch.nn import (
    Conv2d,
    MaxPool2d,
    ReLU,
    LazyLinear,
    Linear,
    Softmax,
    Dropout,
    BatchNorm2d,
    Sequential,
)
from torch import flatten


class Network(nn.Module):

    def __init__(self, start_channels, classes):
        super(Network, self).__init__()
        self.layer1 = Sequential(
            Conv2d(start_channels, 96, kernel_size=11, stride=4, padding=0),
            BatchNorm2d(96),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer2 = Sequential(
            Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            BatchNorm2d(256),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer3 = Sequential(
            Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(384),
            ReLU(),
        )
        self.layer4 = Sequential(
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(384),
            ReLU(),
        )
        self.layer5 = Sequential(
            Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc = Sequential(Dropout(0.5), LazyLinear(4096), ReLU())
        self.fc1 = Sequential(Dropout(0.5), Linear(4096, 4096), ReLU())
        # self.fc = Sequential(LazyLinear(4096), ReLU())
        # self.fc1 = Sequential(Linear(4096, 4096), ReLU())
        self.fc2 = Sequential(Linear(4096, classes))

    def forward(self, batch):
        """Forward function."""
        x = self.layer1(batch)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = flatten(x, 1)
        x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def predict(self, batch):
        return Softmax(dim=1)(self.forward(batch))
