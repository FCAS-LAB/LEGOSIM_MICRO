# Project environment
# SIMULATOR_ROOT, defined by setup_env.sh
BENCHMARK_ROOT=$(SIMULATOR_ROOT)/benchmark/UCIe_PCIe

# Compiler environment of C/C++
CC=g++
CFLAGS=-Wall -Werror -g -I$(SIMULATOR_ROOT)/interchiplet/includes
INTERCHIPLET_C_LIB=$(SIMULATOR_ROOT)/interchiplet/lib/libinterchiplet_c.a

# C/C++ Source file
C_SRCS=net.cpp
C_OBJS=obj/net.o
C_TARGET=bin/net_c

# Compiler environment of CUDA
NVCC=nvcc
CUFLAGS=--compiler-options -Wall -I$(SIMULATOR_ROOT)/interchiplet/includes

# CUDA Source file
CUDA_SRCS=matmul.cu
CUDA_OBJS=cuobj/matmul.o
CUDA_TARGET=bin/matmul_cu

DSA_SRCS=dsa.cpp
DSA_OBJS=obj/dsa.o
DSA_TARGET=bin/dsa

DRAM_SRCS=DRAM.cpp
DRAM_OBJS=obj/DRAM.o
DRAM_TARGET=bin/dram

all: bin_dir obj_dir cuobj_dir C_target DSA_target DRAM_target

# C language target
C_target: $(C_OBJS)
	$(CC) $(C_OBJS) $(INTERCHIPLET_C_LIB) -o $(C_TARGET)

# CUDA language target
CUDA_target: $(CUDA_OBJS)
	$(NVCC) -L$(SIMULATOR_ROOT)/gpgpu-sim/lib/$(GPGPUSIM_CONFIG) --cudart shared $(CUDA_OBJS) -o $(CUDA_TARGET)

# DSA language target
DSA_target: $(DSA_OBJS)
	$(CC) $(DSA_OBJS) $(INTERCHIPLET_C_LIB) -o $(DSA_TARGET)

# DRAM language target
DRAM_target: $(DRAM_OBJS)
	$(CC) $(DRAM_OBJS) $(INTERCHIPLET_C_LIB) -o $(DRAM_TARGET)

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
	../../interchiplet/bin/interchiplet ./net.yml -w 6 -f 32
# Clean generated files.
clean:
	rm -rf bench.txt delayInfo.txt buffer* message_record.txt
	rm -rf proc_r*_t*
	

clean_all:clean
	rm -rf obj cuobj bin