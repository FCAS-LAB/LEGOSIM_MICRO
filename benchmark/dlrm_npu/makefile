# Project environment
# SIMULATOR_ROOT, defined by setup_env.sh
BENCHMARK_ROOT=$(SIMULATOR_ROOT)/benchmark/dlrm_npu

# Compiler environment of C/C++
CC=g++
CFLAGS=-Wall -Werror -g -I$(SIMULATOR_ROOT)/interchiplet/includes
INTERCHIPLET_C_LIB=$(SIMULATOR_ROOT)/interchiplet/lib/libinterchiplet_c.a

# C/C++ Source file
C_SRCS=dlrm.cpp
C_OBJS=obj/dlrm.o
C_TARGET=bin/dlrm

# Compiler environment of CUDA
# NVCC=nvcc
# CUFLAGS=--compiler-options -Wall -I$(SIMULATOR_ROOT)/interchiplet/includes

# CUDA Source file
# CUDA_SRCS=matmul.cu
# CUDA_OBJS=cuobj/matmul.o
# CUDA_TARGET=bin/matmul_cu

# CIM_SRCS=cim.cpp
# CIM_OBJS=obj/cim.o
# CIM_TARGET=bin/cim

NPU_SRCS=npu.cpp
NPU_OBJS=obj/npu.o
NPU_TARGET=bin/npu

all: bin_dir obj_dir C_target NPU_target

# C language target
C_target: $(C_OBJS)
	$(CC) $(C_OBJS) $(INTERCHIPLET_C_LIB) -o $(C_TARGET)

# CUDA language target
# CUDA_target: $(CUDA_OBJS)
# 	$(NVCC) -L$(SIMULATOR_ROOT)/gpgpu-sim/lib/$(GPGPUSIM_CONFIG) --cudart shared $(CUDA_OBJS) -o $(CUDA_TARGET)

# CIM language target
# CIM_target: $(CIM_OBJS)
# 	$(CC) $(CIM_OBJS) $(INTERCHIPLET_C_LIB) -o $(CIM_TARGET)

# NPU language target
NPU_target: $(NPU_OBJS)
	$(CC) $(NPU_OBJS) $(INTERCHIPLET_C_LIB) -o $(NPU_TARGET)

# Rule for C object
obj/%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Rule for Cuda object
cuobj/%.o: %.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@

# Directory for binary files.
bin_dir:
	mkdir -p bin

# Directory for object files for C.
obj_dir:
	mkdir -p obj

# Directory for object files for CUDA.
cuobj_dir:
	mkdir -p cuobj

run:
	../../interchiplet/bin/interchiplet ./dlrm.yml -w 4
# Clean generated files.
clean:
	rm -rf bench.txt delayInfo.txt buffer* message_record.txt
	rm -rf proc_r*_t* *.log
	

clean_all:clean
	rm -rf obj cuobj bin