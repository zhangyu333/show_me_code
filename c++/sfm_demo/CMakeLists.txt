cmake_minimum_required(VERSION 3.21)
project(sfm_demo)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)

set(CMAKE_PREFIX_PATH "/opt/homebrew/Cellar/qt@5/5.15.13_1/lib/cmake/Qt5")
find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
find_package(jsoncpp CONFIG REQUIRED)

add_executable(${PROJECT_NAME} main.cpp my_matcher.cpp my_matcher.h multi_view_3D_reconstruction.cpp multi_view_3D_reconstruction.h)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES})