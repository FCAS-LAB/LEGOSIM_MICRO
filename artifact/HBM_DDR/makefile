# Project environment
# SIMULATOR_ROOT, defined by setup_env.sh
BENCHMARK_ROOT=$(SIMULATOR_ROOT)/benchmark/HBM_DDR
INTERCHIPLET=../../interchiplet/bin/interchiplet

# Compiler environment of C/C++
CC=g++
CFLAGS=-Wall -Werror -g -I$(SIMULATOR_ROOT)/interchiplet/includes
LDFLAGS=-lrt -lpthread
INTERCHIPLET_C_LIB=$(SIMULATOR_ROOT)/interchiplet/lib/libinterchiplet_c.a

# C/C++ Source file
C_SRCS=storage.cpp
C_OBJS=obj/storage.o
C_TARGET=bin/storage

HBM_SRCS=HBM.cpp
HBM_OBJS=obj/HBM.o
HBM_TARGET=bin/HBM

DDR_SRCS=DDR.cpp
DDR_OBJS=obj/DDR.o
DDR_TARGET=bin/DDR

# Compiler environment of CUDA
NVCC=nvcc
CUFLAGS=--compiler-options -Wall -I$(SIMULATOR_ROOT)/interchiplet/includes/
INTERCHIPLET_CU_LIB=$(SIMULATOR_ROOT)/interchiplet/lib/libinterchiplet_cu.a

# CUDA Source file
# CUDA_SRCS = kernel.cu bfs.cu
# CUDA_OBJS = $(patsubst %.cu, cuobj/%.o, $(CUDA_SRCS))
# CUDA_TARGET = bin/bfs_cu

all: bin_dir obj_dir C_target HBM_target DDR_target

# C language target
C_target: $(C_OBJS)
	$(CC) $(C_OBJS) $(INTERCHIPLET_C_LIB) $(LDFLAGS) -o $(C_TARGET) -pthread

# HBM language target
HBM_target: $(HBM_OBJS)
	$(CC) $(HBM_OBJS) $(INTERCHIPLET_C_LIB) -o $(HBM_TARGET)

# DDR language target
DDR_target: $(DDR_OBJS)
	$(CC) $(DDR_OBJS) $(INTERCHIPLET_C_LIB) -o $(DDR_TARGET)

# CUDA language target
# CUDA_target: $(CUDA_OBJS)
# 	$(NVCC) -L$(SIMULATOR_ROOT)/gpgpu-sim/lib/$(GPGPUSIM_CONFIG) -g --cudart shared $(CUDA_OBJS) -o $(CUDA_TARGET)

# Rule for C object
obj/%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Rule for Cuda object
# cuobj/%.o: %.cu 
# 	$(NVCC) $(CUFLAGS) -c $< -o $@

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
	$(INTERCHIPLET) storage.yml -w 3 -f 2

gdb:
	cuda-gdb $(CUDA_TARGET)

# Clean generated files.
clean:
	rm -rf bench.txt delayInfo.txt buffer* message_record.txt
	rm -rf proc_r*_t* 
	rm -rf obj cuobj bin