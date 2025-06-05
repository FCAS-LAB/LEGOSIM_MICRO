cmake_minimum_required(VERSION 3.10)
project(DLRM_REAl LANGUAGES CXX CUDA)

set(CMAKE_PREFIX_PATH "~/libtorch")  # LibTorch路径
find_package(Torch REQUIRED)
set(CMAKE_C_COMPILER "/home/qc/gcc-9.4.0/bin/gcc")
set(CMAKE_CXX_COMPILER "/home/qc/gcc-9.4.0/bin/g++")
set(CMAKE_CXX_STANDARD 17) 

add_executable(dlrm_real dlrm_real.cu)

target_link_libraries(dlrm_real PRIVATE ${TORCH_LIBRARIES}  pthread)

set(CMAKE_BUILD_TYPE Debug)