cmake_minimum_required(VERSION 2.8.3)
project(lidar_super_resolution)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge 
  image_transport
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(octomap REQUIRED)


catkin_package(

)

###########
## Build ##
###########

set(Eigen_INCLUDE_DIRS "/usr/include/eigen3")
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(vlp16_to_range_img src/vlp16_to_range_img.cpp)
target_link_libraries(vlp16_to_range_img
${catkin_LIBRARIES}
${PCL_LIBRARIES}
${OpenCV_LIBS}
)

