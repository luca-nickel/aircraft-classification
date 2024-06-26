import torch
from torch import nn


class CnnModelBoundingBoxes(nn.Module):

    def __init__(self, input_size, anz_channels=3):
        super(CnnModelBoundingBoxes, self).__init__()
        kernel_size = 3
        out_channels = 8
        self.conv_layer1 = nn.Conv2d(
            in_channels=anz_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=1,
            dilation=1,
            stride=1,
        )
        self.conv_layer2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * 2,
            kernel_size=kernel_size,
            padding=1,
            dilation=1,
            stride=1,
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=kernel_size - 1, stride=2)

        self.conv_layer3 = nn.Conv2d(
            in_channels=out_channels * 2,
            out_channels=out_channels * 2 * 2,
            kernel_size=kernel_size,
            padding=1,
            dilation=1,
        )
        self.conv_layer4 = nn.Conv2d(
            in_channels=out_channels * 2 * 2,
            out_channels=out_channels * 2 * 2 * 2,
            kernel_size=kernel_size,
            padding=1,
            dilation=1,
        )
        self.max_pool2 = nn.MaxPool2d(kernel_size=kernel_size - 1, stride=2)
        self.fc1 = nn.Linear(
            # 1048576, 256  # für 512x512 Bilder
            640000, 256  # für 400x400 Bilder
        )
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 4)

    # Progresses data across layers
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
        out = out.reshape(out.size(0), -1)
        # print("OUT  ")
        # print(out.shape)
        # print(out.size())
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
