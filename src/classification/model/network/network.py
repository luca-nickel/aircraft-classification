import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, ReLU, LazyLinear, Linear, Softmax
from torch import flatten


class Network(nn.Module):

    def __init__(self, start_channels, classes):
        super(Network, self).__init__()
        self.conv1 = Conv2d(
            in_channels=start_channels, out_channels=20, kernel_size=(5, 5)
        )
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = LazyLinear(out_features=500)
        self.relu3 = ReLU()
        self.fc2 = Linear(in_features=500, out_features=classes)

    def forward(self, batch):
        """Forward function."""
        x = self.conv1(batch)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

    def predict(self, batch):
        return Softmax(dim=1)(self.forward(batch))
