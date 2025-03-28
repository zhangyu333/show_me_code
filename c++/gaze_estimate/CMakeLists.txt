cmake_minimum_required(VERSION 3.21)
project(gaze_estimate)

find_package(OpenCV)

set(CMAKE_CXX_STANDARD 14)

include_directories(/Users/zhangyu/3rdparty/onnxruntime-osx-arm64-1.17.3/include)
link_directories(/Users/zhangyu/3rdparty/onnxruntime-osx-arm64-1.17.3/lib)

add_executable(${PROJECT_NAME} main.cpp gaze_estimate.hpp face_detect.hpp main.hpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} onnxruntime)