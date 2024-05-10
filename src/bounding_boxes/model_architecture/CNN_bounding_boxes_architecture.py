import torch
from torch import nn


class CNN_model_bounding_boxes(nn.Module):

    def __init__(self, input_size, anz_channals=3):
        super(CNN_model_bounding_boxes, self).__init__()
        kernel_size = 3
        OUT_CHANNELS = 8
        self.conv_layer1 = nn.Conv2d(in_channels=anz_channals, out_channels=OUT_CHANNELS, kernel_size=kernel_size,
                                     padding=1, dilation=1, stride=1)
        self.conv_layer2 = nn.Conv2d(in_channels=OUT_CHANNELS, out_channels=OUT_CHANNELS * 2, kernel_size=kernel_size,
                                     padding=1, dilation=1, stride=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=kernel_size - 1, stride=2)

        self.conv_layer3 = nn.Conv2d(in_channels=OUT_CHANNELS * 2, out_channels=OUT_CHANNELS * 2 * 2,
                                     kernel_size=kernel_size, padding=1, dilation=1)
        self.conv_layer4 = nn.Conv2d(in_channels=OUT_CHANNELS * 2 * 2, out_channels=OUT_CHANNELS * 2 * 2 * 2,
                                     kernel_size=kernel_size, padding=1, dilation=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=kernel_size - 1, stride=2)

        finalOutChannels = OUT_CHANNELS * 2 * 2 * 2
        new_img_dimensions = int(int(input_size / 2) / 2)
        self.fc1 = nn.Linear(finalOutChannels * new_img_dimensions * new_img_dimensions, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 4)

    # Progresses data across layers
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
