cmake_minimum_required(VERSION 3.10)
project(Transformer LANGUAGES CXX)

# Set base paths
set(CMAKE_PREFIX_PATH "~/libtorch-cpu/libtorch")  # LibTorch path
set(SIMULATOR_ROOT $ENV{SIMULATOR_ROOT})
set(BENCHMARK_ROOT ${SIMULATOR_ROOT}/artifact/auto_transformer)
add_subdirectory(third_party/CLI11)

# Print paths for debugging
message(STATUS "SIMULATOR_ROOT = ${SIMULATOR_ROOT}")
message(STATUS "BENCHMARK_ROOT = ${BENCHMARK_ROOT}")

# Set compilation options
set(CMAKE_CXX_STANDARD 17)  # Use C++17 for PyTorch compatibility
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")
set(CMAKE_C_COMPILER "/home/qc/gcc-9.4.0/bin/gcc")
set(CMAKE_CXX_COMPILER "/home/qc/gcc-9.4.0/bin/g++")

# Add include directories
include_directories(${SIMULATOR_ROOT}/interchiplet/includes)
include_directories(${SIMULATOR_ROOT}/snipersim/include)

# Set C++ library paths
set(INTERCHIPLET_C_LIB "${SIMULATOR_ROOT}/interchiplet/lib/libinterchiplet_c.a")
set(SNIPERSIM_LIB "${SIMULATOR_ROOT}/snipersim/lib/libcarbon_sim.a")

# Set output directories - matching makefile structure
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Use CPU version for PyTorch
set(USE_CUDA OFF CACHE BOOL "Disable CUDA for PyTorch" FORCE)
set(USE_CUDNN OFF CACHE BOOL "Disable CUDNN for PyTorch" FORCE)
find_package(Torch REQUIRED)

# C++ PyTorch target
add_executable(Transformer transformer.cpp)
target_link_libraries(Transformer ${TORCH_LIBRARIES} ${INTERCHIPLET_C_LIB} pthread)
target_include_directories(Transformer PUBLIC third_party/CLI11/include)

# add_executable(dram dram.cpp)
# target_link_libraries(dram ${INTERCHIPLET_C_LIB} pthread)
# target_include_directories(dram PUBLIC ${SIMULATOR_ROOT}/interchiplet/includes)

add_executable(dsa dsa.cpp)
target_link_libraries(dsa ${INTERCHIPLET_C_LIB} pthread)
target_include_directories(dsa PUBLIC ${SIMULATOR_ROOT}/interchiplet/includes)

add_executable(mnsim mnsim.cpp)
target_link_libraries(mnsim ${INTERCHIPLET_C_LIB} pthread)
target_include_directories(mnsim PUBLIC ${SIMULATOR_ROOT}/interchiplet/includes)

# Add custom commands to compile CUDA files - simulating makefile approach
# First create necessary directories
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/cuobj
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/cuobj
    COMMENT "Creating CUDA object directory"
)

# Compile CUDA object files
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/cuobj/matmul.o
    DEPENDS ${CMAKE_BINARY_DIR}/cuobj matmul.cu
    COMMAND nvcc --compiler-options -Wall -I${SIMULATOR_ROOT}/interchiplet/includes -c ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cu -o ${CMAKE_BINARY_DIR}/cuobj/matmul.o
    COMMENT "Compiling CUDA object"
)

# Link CUDA executable
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/bin/matmul_cu
    DEPENDS ${CMAKE_BINARY_DIR}/cuobj/matmul.o
    COMMAND nvcc -L${SIMULATOR_ROOT}/gpgpu-sim/lib/$ENV{GPGPUSIM_CONFIG} --cudart shared ${CMAKE_BINARY_DIR}/cuobj/matmul.o -o ${CMAKE_BINARY_DIR}/bin/matmul_cu
    COMMENT "Linking CUDA executable"
)

# Add custom target depending on CUDA executable
add_custom_target(matmul_cu ALL DEPENDS ${CMAKE_BINARY_DIR}/bin/matmul_cu)

# Add run target to simulate makefile run
add_custom_target(run
    COMMAND ${SIMULATOR_ROOT}/interchiplet/bin/interchiplet ${CMAKE_CURRENT_SOURCE_DIR}/auto_transformer.yml -w 6 -f 2000
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running auto_transformer with interchiplet"
)

# Add clean target
add_custom_target(clean_target
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/obj
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/cuobj
    COMMAND sh -c "rm -rf ${CMAKE_BINARY_DIR}/proc_* || true"
    COMMAND sh -c "rm -rf bench.txt delayInfo.txt buffer* message_record.txt || true"
    COMMENT "Cleaning all generated files"
)

# Set debug mode
set(CMAKE_BUILD_TYPE Debug)