/*
 * Created on Fri May 01 2020
 *
 * Copyright (c) 2020 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc

 */

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <velodyne_pointcloud/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <python3.5m/Python.h>

using namespace std;
using namespace pcl;
using namespace cv;

const int kImgHeight = 16;
const int kImgWidth = 1024;

ros::Publisher vlp16_img_pub;

void Callback(const sensor_msgs::PointCloud2::ConstPtr& point_cloud_msg)
{
    // ROS_INFO("Enter callback");

    const float kRadPerCol = 2 * M_PI / kImgWidth;

    Mat range_img(kImgHeight, kImgWidth, CV_32FC1, Scalar(0));

    PointCloud<velodyne_pointcloud::PointXYZIR> cur_cloud;
    fromROSMsg(*point_cloud_msg, cur_cloud);
    for(int i = 0; i < cur_cloud.size(); i++)
    {
        velodyne_pointcloud::PointXYZIR pt = cur_cloud[i];
        if(isnan(pt.x) || isnan(pt.y) || isnan(pt.z))
            continue;
        
        float radius = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
        float omega_rad = -asin(pt.z / radius);
        float alpha_cos_val = pt.x / radius / cos(omega_rad);
        
        if(alpha_cos_val >= 1)
            alpha_cos_val = 0.999999;
        if(alpha_cos_val <= -1)
            alpha_cos_val = -0.999999;
        float alpha_rad = acos(alpha_cos_val);

        if(pt.y > 0)
            alpha_rad = 2 * M_PI - alpha_rad;

        int col_index = (int)round(alpha_rad / kRadPerCol);
        col_index = (col_index + kImgWidth/ 2) % kImgWidth;
        int row_index = pt.ring; 

        float& cur_val = range_img.at<float>(row_index, col_index);
        if(pt.x <  cur_val || cur_val < 0.001f)
        cur_val = pt.x;
    }
    // normalized the image.
    range_img = range_img / 100.f;
    cv_bridge::CvImage cv_range_img(point_cloud_msg->header, "32FC1", range_img);
    sensor_msgs::ImagePtr range_img_msg = cv_range_img.toImageMsg();
    vlp16_img_pub.publish(range_img_msg);
}

int main(int argc, char** argv) {
    // Init python modules.
    Py_Initialize();     
    ros::init(argc, argv, "vlp16_to_range_img.cpp");
    ros::NodeHandle nh("");
    ros::Subscriber vlp16_sub = nh.subscribe("/velodyne_points", 1, &Callback);

    vlp16_img_pub = nh.advertise<sensor_msgs::Image>("/vlp16_range_img", 1);
    ros::spin();
    return 0;
    
}