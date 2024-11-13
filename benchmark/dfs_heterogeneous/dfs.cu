#include <iostream>
#include <queue>
#include <complex>
#include <vector>
#include <cmath>
#include "apis_cu.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NUM_NODES 10

using namespace std;


__global__ void dfsKernel(int *subgraph, bool *visited, int *stack, int startNode, int numNodes) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stackIndex = 0;

    // 初始化栈，将起始节点放入栈中
    if (tid == 0) {
        *stack = startNode;
        visited[startNode] = true;
    }

    __syncthreads();

    // 每个线程都尝试从栈中取出一个节点进行访问
    while (stackIndex < numNodes) {
        int currentNode = -1;

        // 尝试从栈中取出一个节点
        if (tid == 0) {
            if (stackIndex < numNodes && stack[stackIndex] != -1) {
                currentNode = stack[stackIndex];
                stackIndex++;
            } else {
                break;
            }
        }

        __syncthreads();

        if (currentNode != -1) {
            // 访问当前节点的所有邻居
            for (int i = 0; i < numNodes; i++) {
                if (subgraph[currentNode * numNodes + i] && !visited[i]) {
                    // 如果邻居节点未被访问，则将其放入栈中
                    if (tid == 0) {
                        stack[stackIndex] = i;
                        visited[i] = true;
                        stackIndex++;
                    }
                    __syncthreads();
                }
            }
        }
        __syncthreads();
    }
}





int main(int argc, char **argv)
{
    int idX = atoi(argv[1]);
    int idY = atoi(argv[2]);
    int *d_subgraph;
    bool *d_visited;
    int *d_stack;// 用于DFS的栈


    cudaMalloc((void **)&d_subgraph, NUM_NODES * NUM_NODES * sizeof(int));
    cudaMalloc((void **)&d_visited, NUM_NODES * sizeof(bool));
    cudaMalloc((void **)&d_stack, NUM_NODES * sizeof(int));
    for (int i = 0; i < NUM_NODES; ++i)
    {
        cudaMemset(d_visited + i, false, sizeof(bool));
    }
    // 接收子图数据
      std::cout << "cuda jieshouqian 1 1-------------------------------------" << std::endl;

    receiveMessage(idX, idY, 2, 2, d_subgraph, NUM_NODES * NUM_NODES * sizeof(int));
          std::cout << "cuda jieshouqqqqqqqq 2 2-------------------------------------" << std::endl;
    int startNode = 0;
    // 调用 CUDA 核函数进行 DFS
    dim3 threadsPerBlock(NUM_NODES);
    dim3 numBlocks(1);
    dfsKernel<<<numBlocks, threadsPerBlock>>>(d_subgraph, d_visited,d_stack, startNode, NUM_NODES);
    cudaDeviceSynchronize();
    // 发送结果
    sendMessage(2, 2, idX, idY, d_visited, NUM_NODES * sizeof(bool));
    cudaFree(d_subgraph);
    cudaFree(d_visited);
    return 0;
}
