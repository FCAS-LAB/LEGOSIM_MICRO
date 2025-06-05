#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <string>
#include <iostream>
#include "apis_cu.h"
#include "device_launch_parameters.h"
#include <vector>
#define mlp_comm_id 0
#define embedding_comm_id 1

// MLP前向传播的CUDA内核
__global__ void linear_relu_forward_kernel(
    float* input,       // 输入数据 [batch_size, in_features]
    float* weight,      // 权重 [out_features, in_features]
    float* bias,        // 偏置 [out_features]
    float* output,            // 输出 [batch_size, out_features]
    int batch_size,           // 批次大小
    int in_features,          // 输入特征数
    int out_features,         // 输出特征数
    bool apply_relu,          // 是否应用ReLU
    bool apply_sigmoid        // 是否应用Sigmoid
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 添加边界检查
    if (row >= batch_size || col >= out_features) {
        return;
    }
    
    float sum = 0.0f;
    for (int i = 0; i < in_features; i++) {
        sum += input[row * in_features + i] * weight[col * in_features + i];
    }
    
    sum += bias[col];
    
    if (apply_relu) {
        sum = fmaxf(sum, 0.0f);  // 使用fmaxf而不是条件判断
    } else if (apply_sigmoid) {
        sum = 1.0f / (1.0f + (-sum));
    }
    
    output[row * out_features + col] = sum;
}

// MLP多层前向传播函数
void mlp_forward_cuda(
    float* input_d,
    float* output_d,
    float** weights,
    float** biases,
    int batch_size,
    int* layer_sizes,
    int num_layers,
    int* activation_types  // 0: 无, 1: ReLU, 2: Sigmoid
) {
    // 分配设备内存
    float  *d_temp;
    std::cout<<"------------------test 6--------------------" << std::endl;
    // 计算最大层大小用于临时缓冲区
    int max_features = 0;
    for (int i = 0; i <= num_layers; i++) {
        if (layer_sizes[i] > max_features) max_features = layer_sizes[i];
    }
    std::cout<<"------------------test 7--------------------" << std::endl;
    // cudaMalloc(&d_input, batch_size * layer_sizes[0] * sizeof(float));
    // cudaMalloc(&d_output, batch_size * layer_sizes[num_layers] * sizeof(float));
    cudaMalloc((void**)&d_temp, batch_size * max_features * sizeof(float));
    std::cout<<"------------------test 8--------------------" << std::endl;
    // 分配并复制权重和偏置
    // for (int i = 0; i < num_layers; i++) {
    //     int in_features = layer_sizes[i];
    //     int out_features = layer_sizes[i+1];
        
    //     cudaMalloc(&d_weights[i], out_features * in_features * sizeof(float));
    //     cudaMalloc(&d_biases[i], out_features * sizeof(float));
        
    //     cudaMemcpy(d_weights[i], weights[i], out_features * in_features * sizeof(float), cudaMemcpyHostToDevice);
    //     cudaMemcpy(d_biases[i], biases[i], out_features * sizeof(float), cudaMemcpyHostToDevice);
    // }
    
    // 复制输入数据
    // cudaMemcpy(d_input, input, batch_size * layer_sizes[0] * sizeof(float), cudaMemcpyHostToDevice);
    
    // 前向传播
    float *cur_input = input_d;
    float *cur_output;
    std::cout<<"------------------test 9--------------------" << std::endl;
    for (int layer = 0; layer < num_layers; layer++) {
        int in_features = layer_sizes[layer];
        int out_features = layer_sizes[layer+1];
        std::cout<<"------------------test 10--------------------" << std::endl;
        // 确定输出缓冲区
        if (layer == num_layers - 1) {
            cur_output = output_d;  // 最后一层输出到最终缓冲区
        } else {
            cur_output = d_temp;    // 中间层输出到临时缓冲区
        }
        std::cout<<"------------------test 11--------------------" << std::endl;
        // 配置内核启动参数
        int block_size = 8;
        dim3 block(block_size, block_size);
        dim3 grid((batch_size + block_size - 1) / block_size, 
                  (out_features + block_size - 1) / block_size);
        
        printf("Layer %d: [%d, %d] -> [%d, %d], activation: %d\n", 
               layer, batch_size, in_features, batch_size, out_features, activation_types[layer]);
        
        // 启动内核
        linear_relu_forward_kernel<<<grid, block>>>(
            cur_input,
            weights[layer],
            biases[layer],
            cur_output,
            batch_size,
            in_features,
            out_features,
            activation_types[layer] == 1,  // ReLU
            activation_types[layer] == 2   // Sigmoid
        );
        
        // cudaDeviceSynchronize();

        
        // 为下一层准备：当前输出变为下一层输入
        if (layer < num_layers - 1) {
            cur_input = cur_output;
        }
    }
    
    // 复制结果回主机
    // cudaMemcpy(output, d_output, batch_size * layer_sizes[num_layers] * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 释放设备内存
    // cudaFree(d_input);
    // cudaFree(d_output);
    cudaFree(d_temp);
    
    // for (int i = 0; i < num_layers; i++) {
        // cudaFree(d_weights[i]);
        // cudaFree(d_biases[i]);
    // }
}

// MLP通信处理函数 - 类似于embedding_comm
void mlp_forward_comm(int src_x, int src_y, int dst_x, int dst_y) {
    // 接收MLP参数
    int64_t params[4]; // batch_size, num_layers, max_layer_size
    int64_t* params_d;
    cudaMalloc((void**)&params_d, 4 * sizeof(int64_t));
    receiveMessage(src_x, src_y, dst_x, dst_y, params_d, 4 * sizeof(int64_t));
    cudaMemcpy(params, params_d, 4 * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaFree(params_d);
    
    int batch_size = params[0];
    int num_layers = params[1];
    int input_size = params[2];
    int output_size = params[3];
    
    // 接收层大小信息
    // int *layer_sizes = new int[num_layers + 1];
    int *layer_sizes_d;
    cudaMalloc((void**)&layer_sizes_d, (num_layers + 1) * sizeof(int));
    receiveMessage(src_x, src_y, dst_x, dst_y, layer_sizes_d, (num_layers + 1) * sizeof(int));
    int *layer_sizes = new int[num_layers + 1];
    cudaMemcpy(layer_sizes, layer_sizes_d, (num_layers + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(layer_sizes_d);
    
    // 接收激活函数类型
    int *activation_types_d;
    cudaMalloc((void**)&activation_types_d, num_layers * sizeof(int));
    receiveMessage(src_x, src_y, dst_x, dst_y, activation_types_d, num_layers * sizeof(int));
    int *activation_types = new int[num_layers];
    cudaMemcpy(activation_types, activation_types_d, num_layers * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(activation_types_d);
    
    // // 分配内存接收权重和偏置
    // float **weights_d;
    // cudaMalloc((void**)&weights_d, num_layers * sizeof(float*));
    // float **biases_d;
    // cudaMalloc((void**)&biases_d, num_layers * sizeof(float*));

    float **weights = new float*[num_layers];
    float **biases = new float*[num_layers];
    
    
    for (int i = 0; i < num_layers; i++) {
        int in_features = layer_sizes[i];
        int out_features = layer_sizes[i+1];
        std::cout<<"------------------test 0--------------------" << std::endl;
        std::cout<<"in_features: "<<in_features<<std::endl;
        std::cout<<"out_features: "<<out_features<<std::endl;
        // float *weight_layer_d; 
        // cudaMalloc((void**)&weight_layer_d, out_features * in_features * sizeof(float));
        // float *bias_layer_d;  
        // cudaMalloc((void**)&bias_layer_d, out_features * sizeof(float));
        // weights[i] = new float[out_features * in_features];
        // biases[i] = new float[out_features];
        cudaMalloc((void**)&weights[i], out_features * in_features * sizeof(float));
        cudaMalloc((void**)&biases[i], out_features * sizeof(float));
        std::cout<<"------------------test 1--------------------" << std::endl;
        // 接收权重和偏置
        receiveMessage(src_x, src_y, dst_x, dst_y, weights[i], out_features * in_features * sizeof(float));
        receiveMessage(src_x, src_y, dst_x, dst_y, biases[i], out_features * sizeof(float));
        std::cout << "------------------test 2--------------------" << std::endl;
        // cudaMemcpy(weights[i], weight_layer_d, out_features * in_features * sizeof(float), cudaMemcpyDeviceToHost);
        // cudaMemcpy(biases[i], bias_layer_d, out_features * sizeof(float), cudaMemcpyDeviceToHost);
        // cudaFree(weight_layer_d);
        // cudaFree(bias_layer_d);
        std::cout<<"------------------test 3--------------------" << std::endl;
    }
    
    // 接收输入数据
    float *input_d;
    cudaMalloc((void**)&input_d, batch_size * input_size * sizeof(float));
    float *output_d;
    cudaMalloc((void**)&output_d, batch_size * output_size * sizeof(float));

    std::cout<<"------------------test 4--------------------" << std::endl;
    // float *input = new float[batch_size * input_size];
    // float *output = new float[batch_size * output_size];
    
    receiveMessage(src_x, src_y, dst_x, dst_y, input_d, batch_size * input_size * sizeof(float));
    // cudaMemcpy(input, input_d, batch_size * input_size * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaFree(input_d);
    std::cout<<"------------------test 5--------------------" << std::endl;
    // 执行MLP前向传播
    mlp_forward_cuda(input_d, output_d, weights, biases, 
                    batch_size, layer_sizes, num_layers, activation_types);
    
    // cudaMemcpy(output_d, output, batch_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    // 发送结果
    sendMessage(dst_x, dst_y, src_x, src_y, output_d, batch_size * output_size * sizeof(float));
    cudaFree(output_d);
    
    // 释放内存
    for (int i = 0; i < num_layers; i++) {
        cudaFree(weights[i]);
        cudaFree(biases[i]);
    }
    
    delete[] weights;
    delete[] biases;
    delete[] layer_sizes;
    // delete[] activation_types;
    // delete[] input;
    // delete[] output;
}


// EmbeddingBag前向传播的纯CUDA实现（不使用torch::Tensor）
__global__ void embeddingbag_forward_kernel(
    const float* weight,           // 权重矩阵指针 [num_embeddings, embedding_dim]
    const int64_t* indices,        // 索引数组指针 [N]
    const int64_t* offsets,        // 偏移数组指针 [B+1]
    const float* per_sample_weights, // 可选样本权重指针 [N]
    float* output,                 // 输出数组指针 [B, embedding_dim]
    int64_t batch_size,            // 批次大小
    int64_t embedding_dim,         // 嵌入维度
    bool has_weights              // 是否使用样本权重
) {
    // 每个线程块处理一个批次样本
    int b = blockIdx.x;
    
    if (b >= batch_size) return;
    
    // 样本的起始和结束索引位置
    int64_t start_idx = offsets[b];
    int64_t end_idx = offsets[b + 1];
    
    // 每个线程处理嵌入向量的一个或多个维度
    for (int d = threadIdx.x; d < embedding_dim; d += blockDim.x) {
        float sum = 0.0f;
        
        // 累加该样本的所有嵌入向量在当前维度上的值
        for (int64_t idx = start_idx; idx < end_idx; idx++) {
            int64_t embedding_idx = indices[idx];
            float w = has_weights ? per_sample_weights[idx] : 1.0f;
            sum += weight[embedding_idx * embedding_dim + d] * w;
        }
        
        // 写入输出
        output[b * embedding_dim + d] = sum;
    }
}

// 高性能版本 - 使用共享内存和warp内规约
template <int BLOCK_SIZE, int EMBEDDING_DIM>
__global__ void embeddingbag_forward_optimized_kernel(
    const float* weight,
    const int64_t* indices,
    const int64_t* offsets,
    const float* per_sample_weights,
    float* output,
    int64_t batch_size,
    int64_t embedding_dim,
    bool has_weights
) {
    __shared__ float partial_sums[BLOCK_SIZE];
    
    int b = blockIdx.x;
    if (b >= batch_size) return;
    
    int64_t start_idx = offsets[b];
    int64_t end_idx = offsets[b + 1];
    int num_indices = end_idx - start_idx;
    
    // 对每个维度进行处理
    for (int d_offset = 0; d_offset < embedding_dim; d_offset += BLOCK_SIZE) {
        int d = d_offset + threadIdx.x;
        float sum = 0.0f;
        
        if (d < embedding_dim) {
            for (int64_t idx = start_idx; idx < end_idx; idx++) {
                int64_t embedding_idx = indices[idx];
                float w = has_weights ? per_sample_weights[idx] : 1.0f;
                sum += weight[embedding_idx * embedding_dim + d] * w;
            }
        }
        
        partial_sums[threadIdx.x] = sum;
        __syncthreads();
        
        // 写回结果
        if (d < embedding_dim) {
            output[b * embedding_dim + d] = partial_sums[threadIdx.x];
        }
        __syncthreads();
    }
}

// 封装函数，启动内核
void embeddingbag_forward_cuda(
    const float* weight,
    const int64_t* indices,
    const int64_t* offsets,
    const float* per_sample_weights,
    float* output,
    int64_t batch_size,
    int64_t num_embeddings,
    int64_t embedding_dim,
    int64_t num_indices
) {
    const bool has_weights = (per_sample_weights != nullptr);
    
    // 选择适当的线程块大小
    const int threads_per_block = min(512, (int)embedding_dim);
    const dim3 blocks(batch_size);
    const dim3 threads(threads_per_block);
    
    // 启动内核
    embeddingbag_forward_kernel<<<blocks, threads>>>(
        weight,
        indices,
        offsets,
        per_sample_weights,
        output,
        batch_size,
        embedding_dim,
        has_weights
    );
    
    // // 检查CUDA错误
    // cudaError_t error = cudaGetLastError();
    // if (error != cudaSuccess) {
    //     printf("CUDA error: %s\n", cudaGetErrorString(error));
    // }
}

void embedding_comm(int src_x, int src_y, int dst_x, int dst_y){
    int64_t sizeList[4];
    int64_t *sizeList_d;
    cudaMalloc((void**)&sizeList_d, 4 * sizeof(int64_t));
    receiveMessage(src_x, src_y, dst_x, dst_y, sizeList_d, 4 * sizeof(int64_t));
    cudaMemcpy(sizeList, sizeList_d, 4 * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaFree(sizeList_d);

    int64_t num_embeddings = sizeList[0];
    int64_t num_indices = sizeList[1];
    int64_t batch_size = sizeList[2];
    int64_t embedding_dim = sizeList[3];

    int64_t weight_size = num_embeddings * embedding_dim * sizeof(float);
    int64_t indices_size = num_indices * sizeof(int64_t);
    int64_t offsets_size = batch_size * sizeof(int64_t);
    // int64_t per_sample_weights_size = num_indices * sizeof(float);
    int64_t output_size = batch_size * embedding_dim * sizeof(float);

    
    std::cout << "weight_size: " << weight_size << std::endl;
    std::cout << "indices_size: " << indices_size << std::endl;
    std::cout << "offsets_size: " << offsets_size << std::endl;
    // std::cout << "per_sample_weights_size: " << per_sample_weights_size << std::endl;
    std::cout << "output_size: " << output_size << std::endl;
    
    float *weight, *output;
    int64_t *indices, *offsets;

    weight = (float*)malloc(weight_size);
    indices = (int64_t*)malloc(indices_size);
    offsets = (int64_t*)malloc(offsets_size);
    // per_sample_weights = (float*)malloc(per_sample_weights_size);
    output = (float*)malloc(output_size);

    receiveMessage(src_x, src_y, dst_x, dst_y, weight, weight_size);
    receiveMessage(src_x, src_y, dst_x, dst_y, indices, indices_size);
    receiveMessage(src_x, src_y, dst_x, dst_y, offsets, offsets_size);
    // receiveMessage(src_x, src_y, dst_x, dst_y, per_sample_weights, per_sample_weights_size);
    std::cout << "------------------test embeddingbag_forward_cuda start--------------------" << std::endl;
    embeddingbag_forward_cuda(weight, indices, offsets, nullptr, output, batch_size, num_embeddings, embedding_dim, num_indices);
    std::cout << "------------------test embeddingbag_forward_cuda end--------------------" << std::endl;
    sendMessage(dst_x, dst_y, src_x, src_y, output, output_size);

    free(weight);
    free(indices);
    free(offsets);
    // free(per_sample_weights);
    free(output);
}


int main(int argc, char *argv[]){
    int src_x = atoi(argv[1]);
    int src_y = atoi(argv[2]);

    int dst_x = atoi(argv[3]);
    int dst_y = atoi(argv[4]);

    bool is_run = true;
    while(is_run){
        int64_t* module_id_d;
        int64_t module_id;
        bool *is_run_d = nullptr;
        cudaMalloc((void**)&module_id_d, sizeof(int64_t));
        cudaMalloc((void**)&is_run_d, sizeof(bool));
        receiveMessage( src_x, src_y,dst_x, dst_y, is_run_d, sizeof(bool));
        cudaMemcpy(&is_run, is_run_d, sizeof(bool), cudaMemcpyDeviceToHost);
        if(!is_run) break;
        receiveMessage(src_x, src_y, dst_x, dst_y, module_id_d, sizeof(int64_t));
        cudaMemcpy(&module_id, module_id_d, sizeof(int64_t), cudaMemcpyDeviceToHost);
        std::cout<<"--------------------------------"<<std::endl;
        std::cout << "module_id: " << module_id << std::endl;
        std::cout<<"--------------------------------"<<std::endl;
        if(module_id == embedding_comm_id){
            embedding_comm(src_x, src_y, dst_x, dst_y);
        }else if(module_id == mlp_comm_id){
            mlp_forward_comm(src_x, src_y, dst_x, dst_y);
        }
        cudaFree(module_id_d);
    }
}