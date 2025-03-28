cmake_minimum_required(VERSION 3.21)
project(aruco_demo)

set(CMAKE_CXX_STANDARD 14)
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/cv/lib/cmake/opencv4)
find_package(OpenCV)

include_directories(${CMAKE_SOURCE_DIR}/3rdparty/aruco/include)
link_directories(${CMAKE_SOURCE_DIR}/3rdparty/aruco/lib)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} aruco)
