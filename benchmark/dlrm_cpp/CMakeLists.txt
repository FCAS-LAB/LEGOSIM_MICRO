cmake_minimum_required(VERSION 3.10)
project(DLRM LANGUAGES CXX)

# set(CMAKE_PREFIX_PATH "~/libtorch-cpu/libtorch")  # LibTorch路径
# find_package(Torch REQUIRED)
set(CMAKE_C_COMPILER "/home/qc/gcc-9.4.0/bin/gcc")
set(CMAKE_CXX_COMPILER "/home/qc/gcc-9.4.0/bin/g++")
set(CMAKE_CXX_STANDARD 17) 
set(SIMULATOR_ROOT $ENV{SIMULATOR_ROOT})
set(BENCHMARK_ROOT ${SIMULATOR_ROOT}/benchmark/dlrm_cpp)

message(STATUS "SIMULATOR_ROOT = ${SIMULATOR_ROOT}")
message(STATUS "BENCHMARK_ROOT = ${BENCHMARK_ROOT}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -g")
# 添加包含目录
include_directories(${SIMULATOR_ROOT}/interchiplet/includes)
include_directories(${SIMULATOR_ROOT}/snipersim/include)
# 设置C++库路径
set(INTERCHIPLET_C_LIB "${SIMULATOR_ROOT}/interchiplet/lib/libinterchiplet_c.a")
set(SNIPERSIM_LIB "${SIMULATOR_ROOT}/snipersim/lib/libcarbon_sim.a")

# 设置输出目录 - 匹配makefile中的结构
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
add_executable(DLRM dlrm.cpp)

target_link_libraries(DLRM PRIVATE ${INTERCHIPLET_C_LIB} ${SNIPERSIM_LIB} pthread)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/cuobj
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/cuobj
    COMMENT "Creating CUDA object directory"
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/cuobj/dlrm_cu.o
    DEPENDS ${CMAKE_BINARY_DIR}/cuobj dlrm.cu
    COMMAND nvcc --compiler-options -I${SIMULATOR_ROOT}/interchiplet/includes -c ${CMAKE_CURRENT_SOURCE_DIR}/dlrm.cu -o ${CMAKE_BINARY_DIR}/cuobj/dlrm_cu.o
    COMMENT "Compiling CUDA object"
)


# 链接CUDA可执行文件
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/bin/dlrm_cu
    DEPENDS ${CMAKE_BINARY_DIR}/cuobj/dlrm_cu.o
    COMMAND nvcc -L${SIMULATOR_ROOT}/gpgpu-sim/lib/$ENV{GPGPUSIM_CONFIG} --cudart shared ${CMAKE_BINARY_DIR}/cuobj/dlrm_cu.o -o ${CMAKE_BINARY_DIR}/bin/dlrm_cu
    COMMENT "Linking CUDA executable"
)

# 添加自定义目标，依赖于CUDA可执行文件
add_custom_target(dlrm_cu ALL DEPENDS ${CMAKE_BINARY_DIR}/bin/dlrm_cu)

# 添加run目标模拟makefile中的run
add_custom_target(run
    COMMAND ${SIMULATOR_ROOT}/interchiplet/bin/interchiplet ${CMAKE_CURRENT_SOURCE_DIR}/auto_dlrm.yml -w 2
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running auto_transformer with interchiplet"
)
message("cmake dir: ${CMAKE_BINARY_DIR}")
# 添加clean目标
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/obj
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/cuobj
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/CMakeCache.txt
    COMMENT "Cleaning all generated files"
)

add_custom_target(clean_target
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/proc_*
    COMMAND ${CMAKE_COMMAND} -E remove -f bench.txt delayInfo.txt buffer* message_record.txt
    COMMAND ${CMAKE_COMMAND} -E remove -f *.log
)

# 设置调试模式
set(CMAKE_BUILD_TYPE Debug)
