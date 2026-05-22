cmake_minimum_required(VERSION 3.10)
project(lane_runtime_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/aarch64-linux-gnu /usr/include /usr/local/include)
find_library(TENSORRT_LIBRARY nvinfer
    HINTS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT headers/library not found. Install JetPack TensorRT development packages.")
endif()

find_library(CUDART_LIBRARY cudart
    HINTS /usr/local/cuda/lib64 /usr/lib/aarch64-linux-gnu)

if(NOT CUDART_LIBRARY)
    message(FATAL_ERROR "CUDA runtime library not found.")
endif()

find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h
    HINTS /usr/local/cuda/include /usr/include)

if(NOT CUDA_INCLUDE_DIR)
    message(FATAL_ERROR "CUDA headers not found.")
endif()

add_executable(lane_runtime
    src/main.cpp
    src/lane_detector.cpp
    src/tensorrt_runner.cpp
)

add_executable(lane_runtime_can
    src/lane_control_main.cpp
    src/lane_detector.cpp
    src/tensorrt_runner.cpp
)

target_include_directories(lane_runtime PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(lane_runtime_can PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(lane_runtime PRIVATE
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARY}
    ${CUDART_LIBRARY}
)

target_link_libraries(lane_runtime_can PRIVATE
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARY}
    ${CUDART_LIBRARY}
)
