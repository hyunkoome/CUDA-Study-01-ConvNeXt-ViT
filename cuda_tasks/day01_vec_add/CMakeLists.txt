cmake_minimum_required(VERSION 3.18)
project(vec_add LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

add_executable(vec_add_runner test_runner.cpp vec_add_kernel.cu)

# ✅ CUDA 아키텍처 명시 (옵션)
set_target_properties(vec_add_runner PROPERTIES
        CUDA_ARCHITECTURES "89"
)
