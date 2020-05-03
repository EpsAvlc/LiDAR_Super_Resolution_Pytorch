#!/usr/bin/env python2
import rospy 
import rosbag

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image
import sensor_msgs.point_cloud2 as pc2

import torch
import sys
sys.path.append('/home/caoming/Projects/lidar_SR_ws/src/LiDAR_super_resolution/lidar_super_resolution/scripts')

from model import *
from cv_bridge import CvBridge
import cv2
import numpy as np

ang_res_x = 360.0/float(1024) # horizontal resolution
ang_res_y = 33.2/float(64 -1) # vertical resolution
ang_start_y = 16.6 # bottom beam angle

class Converter:
    def __init__(self):
        self.model = UNet()
        self.model.load_state_dict(torch.load('/home/caoming/Projects/lidar_SR_ws/src/LiDAR_super_resolution/models/sr_99.pkl'))
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(self.device)
        self.bridge = CvBridge()

        self.rowList = []
        self.colList = []
        for i in range(64):
            self.rowList = np.append(self.rowList, np.ones(1024)*i)
            self.colList = np.append(self.colList, np.arange(1024))
        self.verticalAngle = np.float32(self.rowList * ang_res_y) - ang_start_y
        self.horizonAngle = - np.float32(self.colList + 1 - (1024/2)) * ang_res_x + 90.0
        self.intensity = self.rowList + self.colList / 1024

        self.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        self.pub = rospy.Publisher("vlp64_points", PointCloud2, queue_size=2)
        self.pub_img = rospy.Publisher("vlp64_range_img", Image, queue_size=2)
        self.sub = rospy.Subscriber("vlp16_range_img",  Image, self.callback, queue_size=2)

    def callback(self, range_img_msg):
        range_image = self.bridge.imgmsg_to_cv2(range_img_msg, desired_encoding='passthrough')
        with torch.no_grad():
            range_img_tensor_16 = torch.from_numpy(range_image)
            range_img_tensor_16 = range_img_tensor_16.unsqueeze(0)
            range_img_tensor_16 = range_img_tensor_16.unsqueeze(0)
            range_img_tensor_16 = range_img_tensor_16.to(self.device)

            range_img_tensor_64 = self.model(range_img_tensor_16)
            range_img_tensor_64 = range_img_tensor_64.cpu()
            range_img_tensor_64 = range_img_tensor_64.squeeze(0)
            range_img_tensor_64 = range_img_tensor_64.squeeze(0)
            range_img_64 = range_img_tensor_64.numpy()
            print(range_img_64.shape)
            range_img_64 = range_img_64 * 100
            self.publishPointCloud(range_img_64, range_img_msg.header)

            range_img_msg_64 = self.bridge.cv2_to_imgmsg(range_img_64, encoding="32FC1")
            self.pub_img.publish(range_img_msg_64)
            
    
    def publishPointCloud(self, rangeImage, header):
        lengthList = rangeImage.reshape(64 * 1024)
        x = np.sin(self.horizonAngle) * np.cos(self.verticalAngle) * lengthList
        y = np.cos(self.horizonAngle) * np.cos(self.verticalAngle) * lengthList
        z = np.sin(self.verticalAngle) * lengthList + 0
        points = np.column_stack((x, y, z, self.intensity))
        vlp64_cloud_msg = pc2.create_cloud(header, self.fields, points)
        self.pub.publish(vlp64_cloud_msg)

rospy.init_node("range_img_to_vlp64")
converter = Converter()
rospy.spin()