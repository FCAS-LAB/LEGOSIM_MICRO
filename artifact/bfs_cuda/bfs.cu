#include<stdio.h>
#include<iostream>
#include<cuda_runtime.h>
#include<stdlib.h>
#include<device_launch_parameters.h>
#include<random>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>
#include <sys/time.h>
#include "apis_cu.h"
#include <thrust/extrema.h>
#include "graph.h"
#include "kernel.h"


void bfs(Node* h_graph_nodes, int* h_graph_edges, bool* h_graph_mask, bool* h_graph_visited, int* h_cost, int no_of_nodes, int edge_list_size, int* d_cost) {
    Node* d_graph_nodes;
    int* d_graph_edges;
    bool* d_graph_mask;
    bool* d_graph_visited;
    bool* d_over;
    bool h_over;

    cudaMalloc(&d_graph_nodes, sizeof(Node) * no_of_nodes);
    cudaMalloc(&d_graph_edges, sizeof(int) * edge_list_size);
    cudaMalloc(&d_graph_mask, sizeof(bool) * no_of_nodes);
    cudaMalloc(&d_graph_visited, sizeof(bool) * no_of_nodes);
    cudaMalloc(&d_over, sizeof(bool));

    cudaMemcpy(d_graph_nodes, h_graph_nodes, sizeof(Node) * no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph_edges, h_graph_edges, sizeof(int) * edge_list_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph_mask, h_graph_mask, sizeof(bool) * no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph_visited, h_graph_visited, sizeof(bool) * no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cost, h_cost, sizeof(int) * no_of_nodes, cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((no_of_nodes + block.x - 1) / block.x);

    do {
        h_over = false;
        cudaMemcpy(d_over, &h_over, sizeof(bool), cudaMemcpyHostToDevice);
        bfsKernel<<<grid, block>>>(d_graph_nodes, d_graph_edges, d_graph_mask, d_graph_visited, d_cost, d_over, no_of_nodes);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_over, d_over, sizeof(bool), cudaMemcpyDeviceToHost);
    } while (h_over);

    cudaMemcpy(h_cost, d_cost, sizeof(int) * no_of_nodes, cudaMemcpyDeviceToHost);

    cudaFree(d_graph_nodes);
    cudaFree(d_graph_edges);
    cudaFree(d_graph_mask);
    cudaFree(d_graph_visited);
    // cudaFree(d_cost);
    cudaFree(d_over);
}

int main(int argc, char **argv) {
    int idX = atoi(argv[1]);
    int idY = atoi(argv[2]);
    int no_of_nodes = 6;
    int edge_list_size = 8;

    int *d_graph_nodes_start;
    int *h_graph_nodes_start = new int[no_of_nodes];
    cudaMalloc((void **)&d_graph_nodes_start, sizeof(int) * no_of_nodes);
    int *d_graph_nodes_no_of_edges;
    int *h_graph_nodes_no_of_edges = new int[no_of_nodes];
    cudaMalloc((void **)&d_graph_nodes_no_of_edges, sizeof(int) * no_of_nodes);

    int *d_graph_edges;
    int *h_graph_edges = new int[edge_list_size];
    cudaMalloc((void **)&d_graph_edges, sizeof(int) * edge_list_size);
    bool *d_graph_mask;
    bool *h_graph_mask = new bool[no_of_nodes];
    cudaMalloc((void **)&d_graph_mask, sizeof(bool) * no_of_nodes);
    bool *d_graph_visited;
    bool *h_graph_visited = new bool[no_of_nodes];
    cudaMalloc((void **)&d_graph_visited, sizeof(bool) * no_of_nodes);
    int *d_cost;
    int *h_cost = new int[no_of_nodes];
    cudaMalloc((void **)&d_cost, sizeof(int) * no_of_nodes);
    receiveMessage(idX, idY, 5, 5, d_graph_nodes_start, no_of_nodes * sizeof(int));
    receiveMessage(idX, idY, 5, 5, d_graph_nodes_no_of_edges, no_of_nodes * sizeof(int));
    receiveMessage(idX, idY, 5, 5, d_graph_edges, edge_list_size * sizeof(int));
    receiveMessage(idX, idY, 5, 5, d_graph_mask, no_of_nodes * sizeof(bool));
    receiveMessage(idX, idY, 5, 5, d_graph_visited, no_of_nodes * sizeof(bool));
    receiveMessage(idX, idY, 5, 5, d_cost, no_of_nodes * sizeof(int));
    cudaMemcpy(h_graph_nodes_start, d_graph_nodes_start, sizeof(int) * no_of_nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_graph_nodes_no_of_edges, d_graph_nodes_no_of_edges, sizeof(int) * no_of_nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_graph_edges, d_graph_edges, sizeof(int) * edge_list_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_graph_mask, d_graph_mask, sizeof(bool) * no_of_nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_graph_visited, d_graph_visited, sizeof(bool) * no_of_nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cost, d_cost, sizeof(int) * no_of_nodes, cudaMemcpyDeviceToHost);

    Node *h_graph_nodes = new Node[no_of_nodes];

    for (int i = 0; i < no_of_nodes; ++i) {
        h_graph_nodes[i].starting = h_graph_nodes_start[i];
        h_graph_nodes[i].no_of_edges = h_graph_nodes_no_of_edges[i];
    }
    bfs(h_graph_nodes, h_graph_edges, h_graph_mask, h_graph_visited, h_cost, no_of_nodes, edge_list_size, d_cost);
    for (int i = 0; i < no_of_nodes; i++) {
        std::cout << "Node " << i << " cost: " << h_cost[i] << std::endl;
    }
    sendMessage(5, 5, idX, idY,  d_cost, no_of_nodes * sizeof(int));

    delete[] h_graph_nodes;
    delete[] h_graph_nodes_start;
    delete[] h_graph_nodes_no_of_edges;
    delete[] h_graph_edges;
    delete[] h_graph_mask;
    delete[] h_graph_visited;
    delete[] h_cost;
    cudaFree(d_graph_nodes_start);
    cudaFree(d_graph_nodes_no_of_edges);
    cudaFree(d_graph_edges);
    cudaFree(d_graph_mask);
    cudaFree(d_graph_visited);
    cudaFree(d_cost);

    return 0;
}