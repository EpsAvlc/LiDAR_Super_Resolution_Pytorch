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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace pcl;
using namespace cv;

const int kImgHeight = 16;
const int kImgWidth = 1024;
void Callback(const sensor_msgs::PointCloud2::ConstPtr& point_cloud_msg)
{
    Mat range_img(kImgWidth, kImgHeight, CV_32FC1, Scalar(0));

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
        
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "vlp16_to_range_img.cpp");
    ros::NodeHandle nh("");
    
    ros::spin();
    return 0;
}