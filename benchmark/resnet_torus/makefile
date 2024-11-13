# Project environment
# SIMULATOR_ROOT, defined by setup_env.sh
BENCHMARK_ROOT=$(SIMULATOR_ROOT)/benchmark/resnet
INTERCHIPLET=../../interchiplet/bin/interchiplet

# Compiler environment of C/C++
CC=g++
CFLAGS=-Wall -Werror -g -I$(SIMULATOR_ROOT)/interchiplet/includes
INTERCHIPLET_C_LIB=$(SIMULATOR_ROOT)/interchiplet/lib/libinterchiplet_c.a

# C/C++ Source file
C_SRCS=resnet.cpp
C_OBJS=obj/resnet.o
C_TARGET=bin/resnet_c

NPU_SRCS=cim.cpp
NPU_OBJS=obj/cim.o
NPU_TARGET=bin/cim

MNSIM_SRCS=mnsim.cpp
MNSIM_OBJS=obj/mnsim.o
MNSIM_TARGET=bin/mnsim

# Compiler environment of CUDA
NVCC=nvcc
CUFLAGS=--compiler-options -Wall -I$(SIMULATOR_ROOT)/interchiplet/includes
INTERCHIPLET_CU_LIB=$(SIMULATOR_ROOT)/interchiplet/lib/libinterchiplet_cu.a

# CUDA Source file
CUDA_SRCS = resnet_block.cu resnet.cu
CUDA_OBJS = $(patsubst %.cu, cuobj/%.o, $(CUDA_SRCS))
CUDA_TARGET = bin/resnet_cu

all: bin_dir obj_dir cuobj_dir C_target CUDA_target NPU_target MNSIM_target

# C language target
C_target: $(C_OBJS)
	$(CC) $(C_OBJS) $(INTERCHIPLET_C_LIB) -o $(C_TARGET) -pthread

# CUDA language target
CUDA_target: $(CUDA_OBJS)
	$(NVCC) -L$(SIMULATOR_ROOT)/gpgpu-sim/lib/$(GPGPUSIM_CONFIG) -g --cudart shared $(CUDA_OBJS) -o $(CUDA_TARGET)

# NPU language target
NPU_target: $(NPU_OBJS)
	$(CC) $(NPU_OBJS) $(INTERCHIPLET_C_LIB) -o $(NPU_TARGET)

# MNSIM language target
MNSIM_target: $(MNSIM_OBJS)
	$(CC) $(MNSIM_OBJS) $(INTERCHIPLET_C_LIB) -o $(MNSIM_TARGET)

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
	$(INTERCHIPLET) resnet.yml

gdb:
	cuda-gdb $(CUDA_TARGET)

# Clean generated files.
clean:
	rm -rf bench.txt delayInfo.txt buffer* message_record.txt
	rm -rf proc_r*_t* *.log
	rm -rf obj cuobj bin