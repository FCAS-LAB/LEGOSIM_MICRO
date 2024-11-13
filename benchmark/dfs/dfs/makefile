# Project environment
# SIMULATOR_ROOT, defined by setup_env.sh
BENCHMARK_ROOT=$(SIMULATOR_ROOT)/benchmark/DFS

# Compiler environment of C/C++
CC=g++
CFLAGS=-Wall -Werror -g -I$(SIMULATOR_ROOT)/interchiplet/includes
INTERCHIPLET_C_LIB=$(SIMULATOR_ROOT)/interchiplet/lib/libinterchiplet_c.a

# C/C++ Source file
C_SRCS=dfs-master.cpp
C_OBJS=obj/dfs-master.o
C_TARGET=bin/dfs-master_c

# C/C++ Source file
Worker_SRCS=dfs-worker.cpp
Worker_OBJS=obj/dfs-worker.o
Worker_TARGET=bin/dfs-worker_c

# Compiler environment of CUDA
NVCC=nvcc
CUFLAGS=--compiler-options -Wall -I$(SIMULATOR_ROOT)/interchiplet/includes

# CUDA Source file
CUDA_SRCS=dfs.cu
CUDA_OBJS=cuobj/dfs.o
CUDA_TARGET=bin/dfs_cu

all: bin_dir obj_dir cuobj_dir C_target Worker_target CUDA_target

# C language target
C_target: $(C_OBJS)
	$(CC) $(C_OBJS) $(INTERCHIPLET_C_LIB) -o $(C_TARGET)

Worker_target: $(Worker_OBJS)
	$(CC) $(Worker_OBJS) $(INTERCHIPLET_C_LIB) -o $(Worker_TARGET)

# CUDA language target
CUDA_target: $(CUDA_OBJS)
	$(NVCC) -L$(SIMULATOR_ROOT)/gpgpu-sim/lib/$(GPGPUSIM_CONFIG) --cudart shared $(CUDA_OBJS) -o $(CUDA_TARGET)

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

# Clean generated files.
clean:
	rm -rf bench.txt delayInfo.txt buffer* message_record.txt
	rm -rf proc_r*_t* *.log
	rm -rf obj cuobj bin
