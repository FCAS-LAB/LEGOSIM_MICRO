cmake_minimum_required(VERSION 3.10)
project(Transformer LANGUAGES CXX)

# 设置基本路径
set(CMAKE_PREFIX_PATH "~/libtorch-cpu/libtorch")  # LibTorch路径
set(SIMULATOR_ROOT $ENV{SIMULATOR_ROOT})
set(BENCHMARK_ROOT ${SIMULATOR_ROOT}/benchmark/auto_transformer)
add_subdirectory(third_party/CLI11)

# 打印路径进行调试
message(STATUS "SIMULATOR_ROOT = ${SIMULATOR_ROOT}")
message(STATUS "BENCHMARK_ROOT = ${BENCHMARK_ROOT}")

# 设置编译选项
set(CMAKE_CXX_STANDARD 17)  # 使用C++17以适配PyTorch
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")
set(CMAKE_C_COMPILER "/home/qc/gcc-9.4.0/bin/gcc")
set(CMAKE_CXX_COMPILER "/home/qc/gcc-9.4.0/bin/g++")

# 添加包含目录
include_directories(${SIMULATOR_ROOT}/interchiplet/includes)

# 设置C++库路径
set(INTERCHIPLET_C_LIB "${SIMULATOR_ROOT}/interchiplet/lib/libinterchiplet_c.a")

# 设置输出目录 - 匹配makefile中的结构
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 对PyTorch使用CPU版本
set(USE_CUDA OFF CACHE BOOL "Disable CUDA for PyTorch" FORCE)
set(USE_CUDNN OFF CACHE BOOL "Disable CUDNN for PyTorch" FORCE)
find_package(Torch REQUIRED)

# C++ PyTorch目标
add_executable(Transformer transformer.cpp)
target_link_libraries(Transformer ${TORCH_LIBRARIES} ${INTERCHIPLET_C_LIB} pthread)
target_include_directories(Transformer PUBLIC third_party/CLI11/include)

# 添加自定义命令来编译CUDA文件 - 模拟makefile方式
# 首先创建必要的目录
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/cuobj
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/cuobj
    COMMENT "Creating CUDA object directory"
)

# 编译CUDA对象文件
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/cuobj/matmul.o
    DEPENDS ${CMAKE_BINARY_DIR}/cuobj matmul.cu
    COMMAND nvcc --compiler-options -Wall -I${SIMULATOR_ROOT}/interchiplet/includes -c ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cu -o ${CMAKE_BINARY_DIR}/cuobj/matmul.o
    COMMENT "Compiling CUDA object"
)

# 链接CUDA可执行文件
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/bin/matmul_cu
    DEPENDS ${CMAKE_BINARY_DIR}/cuobj/matmul.o
    COMMAND nvcc -L${SIMULATOR_ROOT}/gpgpu-sim/lib/$ENV{GPGPUSIM_CONFIG} --cudart shared ${CMAKE_BINARY_DIR}/cuobj/matmul.o -o ${CMAKE_BINARY_DIR}/bin/matmul_cu
    COMMENT "Linking CUDA executable"
)

# 添加自定义目标，依赖于CUDA可执行文件
add_custom_target(matmul_cu ALL DEPENDS ${CMAKE_BINARY_DIR}/bin/matmul_cu)

# 添加run目标模拟makefile中的run
add_custom_target(run
    COMMAND ${SIMULATOR_ROOT}/interchiplet/bin/interchiplet ${CMAKE_CURRENT_SOURCE_DIR}/auto_transformer.yml -w 2
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running auto_transformer with interchiplet"
)

# 添加clean目标
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/obj
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/cuobj
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/proc_*
    COMMAND ${CMAKE_COMMAND} -E remove -f bench.txt delayInfo.txt buffer* message_record.txt
    COMMAND ${CMAKE_COMMAND} -E remove -f *.log
    COMMENT "Cleaning all generated files"
)

# 设置调试模式
set(CMAKE_BUILD_TYPE Debug)