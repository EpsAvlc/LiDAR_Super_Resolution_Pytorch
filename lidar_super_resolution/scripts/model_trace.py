#!/usr/bin/python3

import torch
from model import *

net = UNet()
net.load_state_dict(torch.load('/home/caoming/Projects/lidar_SR_ws/src/LiDAR_super_resolution/models/sr_99.pkl'))
net.eval()
example = torch.rand(1, 1, 16, 1024)

traced_script_module = torch.jit.script(net)
traced_script_module.save('/home/caoming/Projects/lidar_SR_ws/src/LiDAR_super_resolution/lidar_super_resolution/model_serialized/UNet.pt')