cmake_minimum_required(VERSION 2.8.3)
project(mir_ndtmapping)

find_package(catkin REQUIRED)

catkin_package(
  CATKIN_DEPENDS
    ndt_fuser
)

install(DIRECTORY ros/launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/ros/launch
)
