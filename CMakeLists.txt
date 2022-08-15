cmake_minimum_required(VERSION 3.5)

project(example)

set(CMAKE_CXX_STANDARD 17)

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")

# TensorRT
set(TENSORRT_ROOT /home/yao/TensorRT-7.2.3.4/)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include/)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNXPARSER})
message(STATUS "Find TensorRT libs: ${TENSORRT_LIBRARY}")

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "Find OpenCV include at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Find OpenCV libraries: ${OpenCV_LIBRARIES}")

include_directories(${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

aux_source_directory(11_self-driving/src/tensorRT/common SRC_LIST5)
aux_source_directory(11_self-driving/src/tensorRT/infer  SRC_LIST6)
aux_source_directory(11_self-driving/src/app-lane SRC_LIST1)
aux_source_directory(11_self-driving/src/app-ldrn SRC_LIST2)
aux_source_directory(11_self-driving/src/app-road SRC_LIST3)
aux_source_directory(11_self-driving/src/app-yolo SRC_LIST4)
# cuda_add_executable(example 06_yolov5_deploy/05_postprocess/01_post_process.cc 06_yolov5_deploy/05_postprocess/gpu_decode.cu)
cuda_add_executable(example 11_self-driving/src/main.cpp ${SRC_LIST1}  ${SRC_LIST2} ${SRC_LIST3}
${SRC_LIST4}
${SRC_LIST5}
${SRC_LIST6}
11_self-driving/src/app-road/road-mask.cu
11_self-driving/src/app-yolo/yolo-decode.cu
)

target_link_libraries(example   ${CUDA_LIBRARIES} 
                                ${TENSORRT_LIBRARY}
                                ${OpenCV_LIBRARIES}
                                # rt
                                cuda
                                yaml-cpp)