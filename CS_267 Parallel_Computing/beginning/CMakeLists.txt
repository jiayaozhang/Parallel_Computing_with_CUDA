CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(beginning)
FIND_PACKAGE(CUDA REQUIRED)
CUDA_ADD_EXECUTABLE(beginning main.cu)
TARGET_LINK_LIBRARIES(beginning)