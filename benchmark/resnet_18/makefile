BENCHMARK_ROOT=$(SIMULATOR_ROOT)/benchmark/resnet18

CC=g++
CFLAGS=-Wall -Werror -g -I$(SIMULATOR_ROOT)/interchiplet/includes
INTERCHIPLET_C_LIB=$(SIMULATOR_ROOT)/interchiplet/lib/libinterchiplet_c.a
SRC_DIR := .
OBJ_DIR := obj
BIN_DIR := bin

# 获取所有的cpp文件列表
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
# 通过将.cpp替换为.o来生成对象文件列表
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))
# 通过将.cpp替换为可执行文件来生成目标列表
TARGETS := $(patsubst $(SRC_DIR)/%.cpp,$(BIN_DIR)/%,$(SRCS))

# 默认目标为所有可执行文件
all: $(TARGETS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR)/%: $(OBJ_DIR)/%.o | $(BIN_DIR)
	$(CC) $< $(INTERCHIPLET_C_LIB) -o $@ -lonnxruntime

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf bench.txt delayInfo.txt buffer* message_record.txt
	rm -rf proc_r*_t* *.log
	rm -rf obj bin