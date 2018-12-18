cmake_minimum_required(VERSION 3.13)
project(calculateDistance)

set(CMAKE_CXX_STANDARD 11)
include_directories(/home/hermit/C3D-v1.1-openblas/include/)
add_definitions(-Wall -DCPU_ONLY)
add_executable(calculateDistance main.cpp)
target_link_libraries(calculateDistance glog
        /usr/local/lib/libopencv_core.so
        /usr/local/lib/libopencv_videoio.so
        /usr/local/lib/libopencv_imgproc.so
        /usr/lib/x86_64-linux-gnu/libprotobuf.so
        /home/hermit/C3D-v1.1-openblas/build/lib/libcaffe.so
        /usr/lib/x86_64-linux-gnu/libboost_system.so
        /usr/lib/x86_64-linux-gnu/libboost_filesystem.so)