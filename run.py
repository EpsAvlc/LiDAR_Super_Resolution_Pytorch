#!/usr/bin/env python3
from lidar_sr_dataset import *
from model import *

from torch.utils.data import DataLoader
import torch.nn
import torch.optim as optim
def train():
    dataset = LiDARSRDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)    

    criterion = nn.L1Loss()
    net = UNet()
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.00001)
    for epoch in range(100):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = net(batch_x.float())
            # loss = criterion(batch_y.float(), output)
            # optimizer.step()



if __name__ == "__main__":
    train()