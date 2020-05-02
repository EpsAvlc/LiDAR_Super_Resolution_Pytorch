#!/usr/bin/env python2
import rospy 
import rosbag

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image

import torch
import sys
sys.path.append('/home/caoming/Projects/lidar_SR_ws/src/LiDAR_super_resolution/lidar_super_resolution/scripts')

from model import *
from cv_bridge import CvBridge
import cv2
class Converter:

    def __init__(self):
        self.sub = rospy.Subscriber("vlp16_range_img",  Image, self.callback, queue_size=1)
        self.model = UNet()
        self.model.load_state_dict(torch.load('/home/caoming/Projects/lidar_SR_ws/src/LiDAR_super_resolution/models/sr_99.pkl'))
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # self.bridge = CvBridge()
        # print("Create bridge")

    def callback(self, range_img_msg):
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(range_img_msg, desired_encoding='passthrough')
        # print(type(cv_image))
        print(type(cv_image))


if __name__ == "__main__":
    rospy.init_node("range_img_to_vlp64")
    converter = Converter()
    rospy.spin()