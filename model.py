#! /usr/bin/python3
import torch
import numpy as numpy
import torch.nn as nn
from model_part import *

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.convTrans1 = nn.ConvTranspose2d(1, 64, kernel_size=(3, 3), 
            stride=(2,1), padding=(1, 1), output_padding=(1, 0))
        self.bn64 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.convTrans2 = nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), 
         stride=(2,1), padding=(1, 1), output_padding=(1, 0))

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

    def forward(self, x):
        # upscailing twice
        x = self.convTrans1(x)
        x = self.relu(self.bn64(x))
        x = self.convTrans2(x)
        x = self.relu(self.bn64(x))

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = nn.Dropout2d(0.25)(x4)

        


if __name__ == "__main__":
    net = UNet()
    null_input = torch.rand((32, 1, 16, 1024))
    net(null_input)