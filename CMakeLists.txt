cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

add_definitions(-std=c++17)

set(CXX_FLAGS "-Wall" "-pedantic")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(camera_fusion)

find_package(OpenCV 4.1 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Executables for exercises
add_executable (yolo_detect_objects src/detect_objects.cpp src/YoloNeuralNetwork.cpp)
target_link_libraries (yolo_detect_objects ${OpenCV_LIBRARIES})