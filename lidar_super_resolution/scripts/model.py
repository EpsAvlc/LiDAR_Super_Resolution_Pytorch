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

        self.doubleConv = DoubleConv(64, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256) 
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.conv1x1 = nn.Conv2d(64, 1, kernel_size=(1,1))

    def forward(self, x):
        # upscailing twice
        x = self.convTrans1(x)
        x = self.relu(self.bn64(x))
        x = self.convTrans2(x)
        x = self.relu(self.bn64(x)) # 64 * 64 * 1024

        x1 = self.doubleConv(x) # 64 * 64 * 1024
        x2 = self.down1(x1) # 128 * 32 * 512
        x3 = self.down2(x2) # 256 * 16 * 256
        x4 = self.down3(x3) # 512 * 8 * 128
        x5 = nn.Dropout2d(0.25)(self.down4(x4)) # 1024 * 4 * 64

        y4 = self.up1(x5, x4) # 512 * 8 * 128
        y3 = self.up2(y4, x3) # 256 * 16 * 256
        y2 = self.up3(y3, x2) # 128 * 32 * 512
        y1 = self.up4(y2, x1) # 64 * 64 * 1024
        y =  self.conv1x1(y1)

        return y


if __name__ == "__main__":
    net = UNet()
    null_input = torch.rand((32, 1, 16, 1024))
    net(null_input)