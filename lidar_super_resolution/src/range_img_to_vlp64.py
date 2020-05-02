#!/usr/bin/env python3
import rospy 
import rosbag

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image

import torch
from scripts import model

class Converter:

    def __init__(self):
        self.sub = rospy.Subscriber("vlp16_range_img",  Image, self.callback, queue_size=1)

    def callback(self, range_img):
        pass

if __name__ == "__main__":
    rospy.init_node("range_img_to_vlp64")
    converter = Converter()
    rospy.spin()