#!/usr/bin/env python2
from lidar_sr_dataset import *
from model import *

from torch.utils.data import DataLoader
import torch.nn
import torch.optim as optim

def train():
    dataset = LiDARSRDataset()
    loader = DataLoader(dataset, batch_size=16, shuffle=True)    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.L1Loss()
    net = UNet()
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.00001)

    from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_dir='/home/caoming/Projects/lidar_SR_pytorch/logs', comment='SR_torch')
    for epoch in range(100):
        total_loss = 0.0
        i = 1
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = net(batch_x.float())
            loss = criterion(batch_y.float(), output)
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item()
            print('[%d, %d]: %f' % (epoch, i, total_loss / i))
            i = i + 1

        writer.add_scalar('Train/Loss', total_loss, epoch)
        torch.save(net.state_dict(), '/home/caoming/Projects/lidar_SR_pytorch/models/sr_' + str(epoch)+'.pkl')


if __name__ == "__main__":
    train()