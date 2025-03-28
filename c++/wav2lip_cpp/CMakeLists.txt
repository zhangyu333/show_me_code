cmake_minimum_required(VERSION 3.21)
project(wav2lip_cpp)

set(CMAKE_CXX_STANDARD 17)

set(PYTHON_EXECUTABLE "/opt/miniconda/envs/py39/bin/python")
set(PYTHON_INCLUDE_DIR "/opt/miniconda/envs/py39/include")
set(PYTHON_LIBRARY "/opt/miniconda/envs/py39/lib/libpython3.9.dylib")
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)

include_directories("/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3")

include_directories(/Users/zhangyu/3rdparty/onnxruntime-osx-arm64-1.17.3/include)
link_directories(/Users/zhangyu/3rdparty/onnxruntime-osx-arm64-1.17.3/lib)

add_executable(${PROJECT_NAME} main.cpp AudioFile.h
        lmsDet.hpp
        audioProcess.h
        model.h
        imageProcess.h
        wavDrive.hpp
        usePy.hpp)

target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} onnxruntime)