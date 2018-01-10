cmake_minimum_required( VERSION 2.8 )
project(SFM-Reconstruction)

set( CMAKE_BUILD_TYPE "Debug")

find_package( OpenCV REQUIRED )
include_directories( $(OpenCV_INCLUDE_DIRS) )

add_library( triangulation SHARED Triangulation.cpp )
add_library(point_cloud SHARED PointCloud.cpp)

add_executable( reconstruct main.cpp )
target_link_libraries( reconstruct ${OpenCV_LIBS} triangulation point_cloud )
