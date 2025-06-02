cmake_minimum_required(VERSION 3.10)
project(vec_add)

find_package(CUDA REQUIRED)
set(CMAKE_CXX_STANDARD 14)

cuda_add_executable(vec_add_runner test_runner.cpp vec_add_kernel.cu)
