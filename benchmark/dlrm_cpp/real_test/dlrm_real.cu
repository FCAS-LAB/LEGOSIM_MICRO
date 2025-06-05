#include <stdio.h>
#include <sys/time.h>
#include <torch/torch.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <memory>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <thread>
#include <cmath>
#include <tuple>

#define mlp_comm_id 0
#define embedding_comm_id 1



// ============ 保持与仿真器一致的CUDA内核 ============
// MLP前向传播的CUDA内核 - 与dlrm.cu完全一致
__global__ void linear_relu_forward_kernel(
    const float* input,       // 输入数据 [batch_size, in_features]
    const float* weight,      // 权重 [out_features, in_features]
    const float* bias,        // 偏置 [out_features]
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
        sum = 1.0f / (1.0f + /*expf*/(-sum));
    }
    
    output[row * out_features + col] = sum;
}

// 添加时间计算辅助函数
double get_time_diff_ms(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
}

// MLP多层前向传播函数 - 与dlrm.cu完全一致，只去掉通信参数
void mlp_forward_gpu(
    const float* input,
    float* output,
    const float** weights,
    const float** biases,
    int batch_size,
    int* layer_sizes,
    int num_layers,
    int* activation_types  // 0: 无, 1: ReLU, 2: Sigmoid
) {
    struct timeval transfer_start, transfer_end, compute_start, compute_end;
    double data_transfer_time = 0.0;
    double gpu_compute_time = 0.0;
    
    // 分配设备内存
    float *d_input, *d_output, *d_temp;
    float **d_weights, **d_biases;
    
    // 计算最大层大小，用于中间结果缓冲区
    int max_in_features = 0;
    int max_out_features = 0;
    for (int i = 0; i < num_layers; i++) {
        if (layer_sizes[i] > max_in_features) max_in_features = layer_sizes[i];
        if (layer_sizes[i+1] > max_out_features) max_out_features = layer_sizes[i+1];
    }
    
    // 开始数据传输计时
    gettimeofday(&transfer_start, NULL);
    
    // 分配设备内存
    cudaMalloc(&d_input, batch_size * layer_sizes[0] * sizeof(float));
    cudaMalloc(&d_output, batch_size * layer_sizes[num_layers] * sizeof(float));
    cudaMalloc(&d_temp, batch_size * max_out_features * sizeof(float));
    
    // 分配权重和偏置的内存指针数组
    cudaMalloc(&d_weights, num_layers * sizeof(float*));
    cudaMalloc(&d_biases, num_layers * sizeof(float*));
    
    // 为每层分配设备内存并复制数据
    float** h_d_weights = new float*[num_layers];
    float** h_d_biases = new float*[num_layers];
    
    for (int i = 0; i < num_layers; i++) {
        int in_features = layer_sizes[i];
        int out_features = layer_sizes[i+1];
        
        cudaMalloc(&h_d_weights[i], out_features * in_features * sizeof(float));
        cudaMalloc(&h_d_biases[i], out_features * sizeof(float));
        
        cudaMemcpy(h_d_weights[i], weights[i], out_features * in_features * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(h_d_biases[i], biases[i], out_features * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    // 复制权重和偏置指针数组
    cudaMemcpy(d_weights, h_d_weights, num_layers * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, h_d_biases, num_layers * sizeof(float*), cudaMemcpyHostToDevice);
    
    // 复制输入数据
    cudaMemcpy(d_input, input, batch_size * layer_sizes[0] * sizeof(float), cudaMemcpyHostToDevice);
    
    gettimeofday(&transfer_end, NULL);
    data_transfer_time += get_time_diff_ms(transfer_start, transfer_end);
    
    // 开始GPU计算计时
    gettimeofday(&compute_start, NULL);
    
    // 前向传播
    float *cur_input = d_input;
    float *cur_output = d_temp;
    
    // 启动内核前检查
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error before kernel: %s\n", cudaGetErrorString(error));
        return;
    }
    
    for (int layer = 0; layer < num_layers; layer++) {
        int in_features = layer_sizes[layer];
        int out_features = layer_sizes[layer+1];
        
        // 确定当前层的输出缓冲区
        if (layer == num_layers - 1) {
            cur_output = d_output;  // 最后一层输出到最终缓冲区
        } else {
            cur_output = d_temp;    // 中间层输出到临时缓冲区
        }
        
        int block_size = 8;
        dim3 block(block_size,block_size);
        dim3 grid((batch_size+block_size-1)/block_size,(out_features+block_size-1)/block_size);
        
        // 启动内核
        linear_relu_forward_kernel<<<grid, block>>>(
            cur_input,
            h_d_weights[layer],
            h_d_biases[layer],
            cur_output,
            batch_size,
            in_features,
            out_features,
            activation_types[layer] == 1,  // ReLU
            activation_types[layer] == 2   // Sigmoid
        );
        
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA kernel error at layer %d: %s\n", layer, cudaGetErrorString(error));
            return;
        }
        
        // 为下一层准备：当前输出变为下一层输入
        if (layer < num_layers - 1) {
            cur_input = cur_output;
        }
    }
    
    gettimeofday(&compute_end, NULL);
    gpu_compute_time = get_time_diff_ms(compute_start, compute_end);
    
    // 开始数据传输计时（回传结果）
    gettimeofday(&transfer_start, NULL);
    
    // 复制结果回主机
    cudaMemcpy(output, d_output, batch_size * layer_sizes[num_layers] * sizeof(float), cudaMemcpyDeviceToHost);
    
    gettimeofday(&transfer_end, NULL);
    data_transfer_time += get_time_diff_ms(transfer_start, transfer_end);
    
    printf("MLP GPU计算时间: %.4f ms, 数据传输时间: %.4f ms\n", gpu_compute_time, data_transfer_time);
    
    // 释放设备内存
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
    
    for (int i = 0; i < num_layers; i++) {
        cudaFree(h_d_weights[i]);
        cudaFree(h_d_biases[i]);
    }
    
    cudaFree(d_weights);
    cudaFree(d_biases);
    delete[] h_d_weights;
    delete[] h_d_biases;
}

// EmbeddingBag前向传播的CUDA内核 - 与dlrm.cu完全一致
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

// 修改embeddingbag_forward_cuda函数，添加GPU计算计时
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
    // struct timeval compute_start, compute_end;
    cudaEvent_t kernel_start, kernel_end;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);
    
    const bool has_weights = (per_sample_weights != nullptr);
    
    // 选择适当的线程块大小
    const int threads_per_block = min(512, (int)embedding_dim);
    const dim3 blocks(batch_size);
    const dim3 threads(threads_per_block);
    
    // 开始GPU计算计时
    // gettimeofday(&compute_start, NULL);
    cudaEventRecord(kernel_start);
    
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
    
    // cudaDeviceSynchronize();
    cudaEventRecord(kernel_end);
    cudaEventSynchronize(kernel_end);
    float kernel_time = 0.0f;
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_end);
    // gettimeofday(&compute_end, NULL);
    // double gpu_compute_time = get_time_diff_ms(compute_start, compute_end);
    
    printf("Embedding GPU计算时间: %.4f ms\n", kernel_time);
    
    // 检查CUDA错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}

// ============ 与dlrm.cpp保持一致的主要类结构 ============
class DLRMImpl : torch::nn::Module{
public:
    int64_t ndevices;
    int64_t output_d;
    int64_t parallel_model_batch_size;
    bool parallel_model_is_not_prepared;
    std::string arch_interaction_op;
    bool arch_interaction_itself;
    bool sync_dense_params;
    float loss_threshold;
    std::string loss_function;
    std::string weighted_pooling;
    bool qr_flag;
    std::string qr_operation;
    int64_t qr_collisions;
    int64_t qr_threshold;
    bool md_flag;
    int64_t md_threshold;
    int64_t n_global_emb;
    int64_t n_local_emb;
    int64_t n_emb_per_rank;
    std::vector<int64_t> local_emb_slice;
    std::vector<int64_t> local_emb_indices;
    torch::nn::ModuleList emb_l{nullptr};
    torch::nn::Sequential bot_l{nullptr}, top_l{nullptr};
    bool quantize_emb;
    int64_t quantize_bits;
    torch::nn::MSELoss loss_fn{nullptr};
    torch::nn::ParameterList v_W_l{nullptr};
    std::vector<torch::nn::Sequential> bot_l_replicas{nullptr}, top_l_replicas{nullptr};

    // 与dlrm.cpp保持一致的create_mlp函数
    torch::nn::Sequential create_mlp(std::vector<int64_t> ln, int64_t sigmoid_layer){
        torch::nn::Sequential layers;
        for(size_t i=0; i<ln.size()-1;i++){
            int64_t n = ln[i];
            int64_t m = ln[i+1];

            auto linear = torch::nn::Linear(torch::nn::LinearOptions(n, m).bias(true));
            float mean = 0.0;
            float std_dev = sqrt(2.0f/(m+n));
            auto W = torch::randn({m, n}) * std_dev + mean;

            std_dev = std::sqrt(1.0f / m);
            auto bt = torch::randn({m}) * std_dev + mean;

            linear->weight.set_data(W.clone().set_requires_grad(true));
            linear->bias.set_data(bt.clone().set_requires_grad(true));

            layers->push_back(linear);

            if(i == sigmoid_layer){
                layers->push_back(torch::nn::Sigmoid());
            }else{
                layers->push_back(torch::nn::ReLU());
            }
        }
        return layers;
    }

    // 与dlrm.cpp保持一致的create_emb函数
    std::tuple<torch::nn::ModuleList, std::vector<torch::Tensor>> create_emb(int64_t m, std::vector<int64_t> ln, std::string weighted_pooling=""){
        torch::nn::ModuleList emb_l;
        emb_l = register_module("emb_l", torch::nn::ModuleList());
        std::vector<torch::Tensor> v_W_l;
        for(size_t i=0; i<ln.size(); i++){
            int64_t n = ln[i];
            torch::nn::EmbeddingBag EE = torch::nn::EmbeddingBag(torch::nn::EmbeddingBagOptions(n,m).mode(torch::kSum).sparse(true));
            float bound = std::sqrt(1.0f / n);
            auto W = torch::empty({n, m}, torch::kFloat32)
                    .uniform_(-bound, bound);
            EE->weight.set_data(W.clone().set_requires_grad(true));
            if(weighted_pooling == ""){
                v_W_l.push_back(torch::Tensor());
            }else{
                v_W_l.push_back(torch::ones(n, torch::kFloat32));
            }
            emb_l->push_back(EE);
        } 
        return {emb_l, v_W_l};
    }

    // 与dlrm.cpp保持一致的interact_features函数
    torch::Tensor interact_features(torch::Tensor x, std::vector<torch::Tensor> ly) {
        torch::Tensor R;
        
        if (arch_interaction_op == "dot") {
            auto batch_size = x.size(0);
            auto d = x.size(1);
            
            std::vector<torch::Tensor> concat_list = {x};
            concat_list.insert(concat_list.end(), ly.begin(), ly.end());
            torch::Tensor T = torch::cat(concat_list, 1).view({batch_size, -1, d});
            
            torch::Tensor T_transpose = T.transpose(1, 2);
            torch::Tensor Z = torch::bmm(T, T_transpose);
            
            auto sizes = Z.sizes();
            int64_t ni = sizes[1];
            int64_t nj = sizes[2];
            
            int64_t offset = arch_interaction_itself ? 1 : 0;
            std::vector<int64_t> li_vec, lj_vec;
            
            for (int64_t i = 0; i < ni; i++) {
                for (int64_t j = 0; j < i + offset; j++) {
                    li_vec.push_back(i);
                    lj_vec.push_back(j);
                }
            }
            
            auto options = torch::TensorOptions().dtype(torch::kLong).device(Z.device());
            torch::Tensor li = torch::tensor(li_vec, options);
            torch::Tensor lj = torch::tensor(lj_vec, options);
            
            torch::Tensor Zflat = Z.index({torch::indexing::Slice(), li, lj});
            R = torch::cat({x, Zflat}, 1);
        } 
        else if (arch_interaction_op == "cat") {
            std::vector<torch::Tensor> concat_list = {x};
            concat_list.insert(concat_list.end(), ly.begin(), ly.end());
            R = torch::cat(concat_list, 1);
        } 
        else {
            throw std::runtime_error("ERROR: --arch-interaction-op=" + arch_interaction_op + " is not supported");
        }
        
        return R;
    }

    // 修改embeddingbag_forward_cuda_wrapper函数，添加数据传输计时
    void embeddingbag_forward_cuda_wrapper(
        float* weight,
        int64_t* indices,
        int64_t* offsets,
        float* per_sample_weights,
        float* output,
        int64_t batch_size,
        int64_t num_embeddings,
        int64_t embedding_dim,
        int64_t num_indices,
        int device_id
    ){
        struct timeval transfer_start, transfer_end;
        double data_transfer_time = 0.0;
        
        // 设置GPU设备
        cudaSetDevice(device_id);
        
        std::cout<<"--------------embeddingbag_forward_cuda start--------------------"<<std::endl;
        
        
        
        // 分配GPU内存
        float *d_weight, *d_output;
        int64_t *d_indices, *d_offsets;
        float *d_per_sample_weights = nullptr;
        
        size_t weight_size = num_embeddings * embedding_dim * sizeof(float);
        size_t indices_size = num_indices * sizeof(int64_t);
        size_t offsets_size = batch_size * sizeof(int64_t);
        size_t output_size = batch_size * embedding_dim * sizeof(float);
        
        cudaMalloc(&d_weight, weight_size);
        cudaMalloc(&d_indices, indices_size);
        cudaMalloc(&d_offsets, offsets_size);
        cudaMalloc(&d_output, output_size);
        
        if (per_sample_weights) {
            cudaMalloc(&d_per_sample_weights, num_indices * sizeof(float));
            gettimeofday(&transfer_start, NULL);
            cudaMemcpy(d_per_sample_weights, per_sample_weights, num_indices * sizeof(float), cudaMemcpyHostToDevice);
            gettimeofday(&transfer_end, NULL);
            data_transfer_time += get_time_diff_ms(transfer_start, transfer_end);
        }
        // 开始数据传输计时
        gettimeofday(&transfer_start, NULL);
        // 复制数据到GPU
        cudaMemcpy(d_weight, weight, weight_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_indices, indices, indices_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_offsets, offsets, offsets_size, cudaMemcpyHostToDevice);
        
        gettimeofday(&transfer_end, NULL);
        data_transfer_time += get_time_diff_ms(transfer_start, transfer_end);
        
        // 调用CUDA内核
        embeddingbag_forward_cuda(d_weight, d_indices, d_offsets, d_per_sample_weights, 
                                 d_output, batch_size, num_embeddings, embedding_dim, num_indices);
        
        // 开始数据传输计时（回传结果）
        gettimeofday(&transfer_start, NULL);
        
        // 复制结果回CPU
        cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
        
        gettimeofday(&transfer_end, NULL);
        data_transfer_time += get_time_diff_ms(transfer_start, transfer_end);
        
        printf("Embedding数据传输时间: %.4f ms\n", data_transfer_time);
        
        // 释放GPU内存
        cudaFree(d_weight);
        cudaFree(d_indices);
        cudaFree(d_offsets);
        cudaFree(d_output);
        if (d_per_sample_weights) {
            cudaFree(d_per_sample_weights);
        }
        
        std::cout<<"--------------embeddingbag_forward_cuda end--------------------"<<std::endl;
    }

    // 修改后的apply_emb函数 - 保持所有数据在CPU上
    std::vector<torch::Tensor> apply_emb(
        const torch::Tensor& lS_o,
        const std::vector<torch::Tensor>& lS_i,
        torch::nn::ModuleList& emb_l,
        torch::nn::ParameterList& v_W_l) {
        
        std::vector<torch::Tensor> ly;
        
        for (size_t k = 0; k < lS_i.size(); k++) {
            torch::Tensor sparse_index_group_batch = lS_i[k].cpu(); // 确保在CPU上
            torch::Tensor sparse_offset_group_batch = lS_o[k].cpu(); // 确保在CPU上
            
            torch::Tensor per_sample_weights;
            if (v_W_l[k].defined()) {
                per_sample_weights = v_W_l[k].index_select(0, sparse_index_group_batch).cpu();
            }
            
            auto E = emb_l[k];
            auto emb_bag = std::dynamic_pointer_cast<torch::nn::EmbeddingBagImpl>(E);
            emb_bag->to(torch::kCPU); // 确保embedding在CPU上
            
            // 使用PyTorch的EmbeddingBag
            // torch::Tensor V;
            // if (per_sample_weights.defined()) {
            //     V = emb_bag->forward(sparse_index_group_batch, sparse_offset_group_batch, per_sample_weights);
            // } else {
            //     V = emb_bag->forward(sparse_index_group_batch, sparse_offset_group_batch);
            // }
            
            // 创建输出张量在CPU上
            torch::Tensor V_cuda = torch::empty({sparse_offset_group_batch.size(0), emb_bag->weight.size(1)}, 
                                    torch::kFloat32); // 默认在CPU
            
            // 调用自定义CUDA函数进行对比/校准，使用指定的GPU
            int device_id = k % this->ndevices;
            if (per_sample_weights.defined()) {
                embeddingbag_forward_cuda_wrapper(
                    emb_bag->weight.data_ptr<float>(),
                    sparse_index_group_batch.data_ptr<int64_t>(), 
                    sparse_offset_group_batch.data_ptr<int64_t>(),
                    per_sample_weights.data_ptr<float>(),
                    V_cuda.data_ptr<float>(),
                    sparse_offset_group_batch.size(0), 
                    emb_bag->weight.size(0),
                    emb_bag->weight.size(1), 
                    sparse_index_group_batch.size(0),
                    device_id);
            } else {
                embeddingbag_forward_cuda_wrapper(
                    emb_bag->weight.data_ptr<float>(),
                    sparse_index_group_batch.data_ptr<int64_t>(), 
                    sparse_offset_group_batch.data_ptr<int64_t>(),
                    nullptr,
                    V_cuda.data_ptr<float>(),
                    sparse_offset_group_batch.size(0), 
                    emb_bag->weight.size(0),
                    emb_bag->weight.size(1), 
                    sparse_index_group_batch.size(0),
                    device_id);
            }
            
            // 返回CPU上的结果 - 可以选择V或V_cuda进行对比
            ly.push_back(V_cuda);  // 使用CUDA版本的结果
        }
        
        return ly;
    }

    // 修改后的mlp_forward_parallel函数 - 保持数据在CPU，只在CUDA调用时移动
    void mlp_forward_parallel(torch::Tensor dense_x_chunk, torch::Tensor& output, int device_id, bool is_top_layer = false) {
        // 设置GPU设备
        cudaSetDevice(device_id % this->ndevices);
        torch::nn::Sequential& mlp = is_top_layer ? top_l : bot_l;
        
        // 确保输入数据在CPU上
        dense_x_chunk = dense_x_chunk.cpu();
        
        // 从Sequential中提取权重和偏置 - 保持与原代码一致
        int num_layers = mlp->size() / 2;
        std::cout<<"num_layers: "<<num_layers<<std::endl;
        
        int64_t params[4];
        params[0] = dense_x_chunk.size(0); // batch_size
        params[1] = num_layers;
        params[2] = dense_x_chunk.size(1); // 输入维度
        
        std::vector<int> layer_sizes(num_layers + 1);
        std::vector<int> activation_types(num_layers);
        
        layer_sizes[0] = dense_x_chunk.size(1);
        
        for (int i = 0; i < num_layers; i++) {
            auto layer = std::dynamic_pointer_cast<torch::nn::LinearImpl>(mlp[i*2]);
            layer_sizes[i+1] = layer->weight.size(0);
            
            auto activation = mlp[i*2+1];
            if (std::dynamic_pointer_cast<torch::nn::ReLUImpl>(activation)) {
                activation_types[i] = 1; // ReLU
            } else if (std::dynamic_pointer_cast<torch::nn::SigmoidImpl>(activation)) {
                activation_types[i] = 2; // Sigmoid
            } else {
                activation_types[i] = 0; // 无激活函数
            }
        }
        
        params[3] = layer_sizes[num_layers];
        
        // 准备权重和偏置数据 - 确保在CPU上
        std::vector<float*> weights_host(num_layers);
        std::vector<float*> biases_host(num_layers);
        
        for (int i = 0; i < num_layers; i++) {
            auto layer = std::dynamic_pointer_cast<torch::nn::LinearImpl>(mlp[i*2]);
            auto weight_contiguous = layer->weight.contiguous().cpu();
            auto bias_contiguous = layer->bias.contiguous().cpu();
            
            weights_host[i] = weight_contiguous.data_ptr<float>();
            biases_host[i] = bias_contiguous.data_ptr<float>();
        }
        
        // 准备输入和输出 - 在CPU上
        auto input_contiguous = dense_x_chunk.contiguous();
        output = torch::empty({dense_x_chunk.size(0), layer_sizes[num_layers]}, torch::kFloat32); // CPU上
        
        // 调用CUDA函数，内部会设置GPU并管理内存
        mlp_forward_gpu(
            input_contiguous.data_ptr<float>(),
            output.data_ptr<float>(),
            (const float**)weights_host.data(),
            (const float**)biases_host.data(),
            params[0], // batch_size
            layer_sizes.data(),
            num_layers,
            activation_types.data()
        );
        
    }

    // 修改mlp_forward_cuda函数 - 添加GPU设备设置
    std::vector<torch::Tensor> mlp_forward_cuda(std::vector<torch::Tensor> dense_x_chunk_list, bool is_top_layer = false) {
        std::vector<torch::Tensor> output(dense_x_chunk_list.size());
        int divide_num = dense_x_chunk_list.size();
        std::vector<std::thread> threads;
        for(int i = 0; i < divide_num; i++){
            torch::Tensor dense_x_chunk = dense_x_chunk_list[i];
            
                     
            // mlp_forward_parallel(dense_x_chunk, output[i], i, is_top_layer);
            threads.push_back(std::thread(std::bind(&DLRMImpl::mlp_forward_parallel, this, dense_x_chunk, std::ref(output[i]), i, is_top_layer)));
        }
        for(auto& thread : threads){
            thread.join();
        }
        return output;
    }

    // 保持与dlrm.cpp一致的其他函数
    std::vector<torch::Tensor> scatter_tensors(
        const torch::Tensor& tensor, 
        const std::vector<int64_t>& device_ids,
        int64_t dim = 0) {
        
        auto outputs_t = torch::chunk(tensor, device_ids.size(), dim);
        std::vector<torch::Tensor> outputs(outputs_t.begin(), outputs_t.end());
        for (size_t i = 0; i < device_ids.size(); i++) {
            torch::Device device(torch::kCUDA, device_ids[i]);
            outputs[i] = outputs[i].to(device);
        }
        return outputs;
    }

    torch::Tensor gather_tensors(
        const std::vector<torch::Tensor>& tensors, 
        int64_t dim = 0) {
        
        std::vector<torch::Tensor> aligned_tensors;
        torch::Device target_device = tensors[0].device();
        
        for (const auto& tensor : tensors) {
            aligned_tensors.push_back(tensor.to(target_device));
        }
        
        return torch::cat(aligned_tensors, dim);
    }

    // 修改parallel_forward_cuda函数 - 确保所有数据在CPU上
    torch::Tensor parallel_forward_cuda(torch::Tensor dense_x, torch::Tensor lS_o, std::vector<torch::Tensor> lS_i) {
        // 确保输入数据在CPU上
        dense_x = dense_x.cpu();
        lS_o = lS_o.cpu();
        for(auto& tensor : lS_i) {
            tensor = tensor.cpu();
        }
        struct timeval start, end;
        gettimeofday(&start, NULL);
        std::cout<<"--------------bot_mlp_forward_cuda start--------------------"<<std::endl;
        auto dense_x_chunk = torch::chunk(dense_x, this->ndevices, 0);
        std::vector<torch::Tensor> dense_x_chunk_list(dense_x_chunk.begin(), dense_x_chunk.end());
        this->ndevices = dense_x_chunk_list.size();
        std::vector<torch::Tensor> x = mlp_forward_cuda(dense_x_chunk_list,false);
        std::cout<<"--------------bot_mlp_forward_cuda end--------------------"<<std::endl;
        gettimeofday(&end, NULL);
        std::cout<<"bot_mlp_forward_cuda time: "<<(end.tv_sec - start.tv_sec)*1000 + (end.tv_usec - start.tv_usec)/1000<<"ms"<<std::endl;
        gettimeofday(&start, NULL);
        std::cout<<"--------------apply_emb start--------------------"<<std::endl;
        std::vector<torch::Tensor> ly = apply_emb(lS_o, lS_i, emb_l, v_W_l);
        std::cout<<"--------------apply_emb end--------------------"<<std::endl;
        gettimeofday(&end, NULL);
        std::cout<<"apply_emb time: "<<(end.tv_sec - start.tv_sec)*1000 + (end.tv_usec - start.tv_usec)/1000<<"ms"<<std::endl;
        gettimeofday(&start, NULL);
        std::cout<<"--------------interact_features start--------------------"<<std::endl;
        std::vector<std::vector<torch::Tensor>> t_list_2d;
        for (size_t k = 0; k < emb_l->size(); k++) {
            auto ly_chunk = torch::chunk(ly[k], this->ndevices, 0);
            std::vector<torch::Tensor> scattered(ly_chunk.begin(), ly_chunk.end());
            t_list_2d.push_back(scattered);
        }
        
        std::vector<std::vector<torch::Tensor>> ly_per_device(ndevices);
        for (size_t i = 0; i < ndevices; i++) {
            ly_per_device[i].resize(emb_l->size());
            for (size_t j = 0; j < emb_l->size(); j++) {
                // 保持在CPU上
                ly_per_device[i][j] = t_list_2d[j][i];
            }
        }
        
        std::vector<torch::Tensor> z;
        for (int64_t k = 0; k < ndevices; k++) {
            // 所有数据都在CPU上
            torch::Tensor x_cpu = x[k];
            
            std::cout<<"x["<<k<<"] device: "<<x_cpu.device()<<std::endl;
            std::cout<<"ly_per_device["<<k<<"][0] device: "<<ly_per_device[k][0].device()<<std::endl;
            
            // 在CPU上执行interact_features
            torch::Tensor z_cpu = interact_features(x_cpu, ly_per_device[k]);
            z.push_back(z_cpu); // 保持在CPU上
        }
        std::cout<<"--------------interact_features end--------------------"<<std::endl;
        gettimeofday(&end, NULL);
        std::cout<<"interact_features time: "<<(end.tv_sec - start.tv_sec)*1000 + (end.tv_usec - start.tv_usec)/1000<<"ms"<<std::endl;
        gettimeofday(&start, NULL);
        std::cout<<"--------------top_mlp_forward_cuda start--------------------"<<std::endl;
        std::vector<torch::Tensor> p = mlp_forward_cuda(z,true);
        std::cout<<"--------------top_mlp_forward_cuda end--------------------"<<std::endl;
        gettimeofday(&end, NULL);
        std::cout<<"top_mlp_forward_cuda time: "<<(end.tv_sec - start.tv_sec)*1000 + (end.tv_usec - start.tv_usec)/1000<<"ms"<<std::endl;
        torch::Tensor p0 = gather_tensors(p, 0);
        return p0;
    }

    torch::Tensor forward(torch::Tensor dense_x, torch::Tensor lS_o, std::vector<torch::Tensor> lS_i){
        if(this->ndevices > 0 ) return parallel_forward_cuda(dense_x, lS_o, lS_i);
        else throw std::runtime_error("Sequential forward not implemented for real hardware");
    }
    
    // 保持与dlrm.cpp一致的构造函数，去掉通信相关参数
    DLRMImpl(
        int64_t m_spa,
        const std::vector<int64_t>& ln_emb,
        const std::vector<int64_t>& ln_bot,
        const std::vector<int64_t>& ln_top,
        const std::string& arch_interaction_op,
        bool arch_interaction_itself,
        int64_t sigmoid_bot,
        int64_t sigmoid_top,
        bool sync_dense_params,
        float loss_threshold,
        int64_t ndevices,
        bool qr_flag,
        const std::string& qr_operation,
        int64_t qr_collisions,
        int64_t qr_threshold,
        bool md_flag,
        int64_t md_threshold,
        const std::string& weighted_pooling,
        const std::string& loss_function) {
        
        if (m_spa != 0 && !ln_emb.empty() && !ln_bot.empty() && 
            !ln_top.empty() && !arch_interaction_op.empty()) {
            
            this->ndevices = ndevices;
            this->output_d = 0;
            this->parallel_model_batch_size = -1;
            this->parallel_model_is_not_prepared = true;
            this->arch_interaction_op = arch_interaction_op;
            this->arch_interaction_itself = arch_interaction_itself;
            this->sync_dense_params = sync_dense_params;
            this->loss_threshold = loss_threshold;
            this->loss_function = loss_function;
            
            if (weighted_pooling != "" && weighted_pooling != "fixed") {
                this->weighted_pooling = "learned";
            } else {
                this->weighted_pooling = weighted_pooling;
            }
            
            this->qr_flag = qr_flag;
            if (this->qr_flag) {
                this->qr_collisions = qr_collisions;
                this->qr_operation = qr_operation;
                this->qr_threshold = qr_threshold;
            }
            
            this->md_flag = md_flag;
            if (this->md_flag) {
                this->md_threshold = md_threshold;
            }
            
            this->v_W_l = register_module("v_W_l", torch::nn::ParameterList());
            
            std::vector<torch::Tensor> w_list;
            std::tie(this->emb_l, w_list) = this->create_emb(m_spa, ln_emb, weighted_pooling);
            
            if (this->weighted_pooling == "learned") {
                for (auto& w : w_list) {
                    this->v_W_l->append(register_parameter("v_W_" + std::to_string(v_W_l->size()), w));
                }
            } else {
                for (auto& w : w_list) {
                    this->v_W_l->append(w);
                }
            }
            
            this->bot_l = register_module("bot_l", this->create_mlp(ln_bot, sigmoid_bot));
            this->top_l = register_module("top_l", this->create_mlp(ln_top, sigmoid_top));
            
            this->quantize_emb = false;
            this->quantize_bits = 32;
            
            this->loss_fn = register_module("loss_fn", torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean)));
        }
    }
};

TORCH_MODULE(DLRM);
// EmbeddingBag前向传播的CUDA内核 - 与dlrm.cu完全一致
// __global__ void embeddingbag_forward_kernel(
//     const float* weight,           // 权重矩阵指针 [num_embeddings, embedding_dim]
//     const int64_t* indices,        // 索引数组指针 [N]
//     const int64_t* offsets,        // 偏移数组指针 [B+1]
//     const float* per_sample_weights, // 可选样本权重指针 [N]
//     float* output,                 // 输出数组指针 [B, embedding_dim]
//     int64_t batch_size,            // 批次大小
//     int64_t embedding_dim,         // 嵌入维度
//     bool has_weights              // 是否使用样本权重
// ) {
//     // 每个线程块处理一个批次样本
//     int b = blockIdx.x;
    
//     if (b >= batch_size) return;
    
//     // 样本的起始和结束索引位置
//     int64_t start_idx = offsets[b];
//     int64_t end_idx = offsets[b + 1];
    
//     // 每个线程处理嵌入向量的一个或多个维度
//     for (int d = threadIdx.x; d < embedding_dim; d += blockDim.x) {
//         float sum = 0.0f;
        
//         // 累加该样本的所有嵌入向量在当前维度上的值
//         for (int64_t idx = start_idx; idx < end_idx; idx++) {
//             int64_t embedding_idx = indices[idx];
//             float w = has_weights ? per_sample_weights[idx] : 1.0f;
//             sum += weight[embedding_idx * embedding_dim + d] * w;
//         }
        
//         // 写入输出
//         output[b * embedding_dim + d] = sum;
//     }
// }
// void test_embeddingbag_forward() {
//     // 1. 测试参数设置
//     const int64_t num_embeddings = 20;    // 嵌入表大小
//     const int64_t embedding_dim = 50;      // 嵌入维度
//     const int64_t batch_size = 10;         // 批次大小
//     const int64_t indices_per_batch = 4;   // 每个批次的索引数量
//     const int64_t num_indices = batch_size * indices_per_batch;  // 总索引数量

//     // 2. 分配主机内存
//     std::vector<float> h_weight(num_embeddings * embedding_dim);
//     std::vector<int64_t> h_indices(num_indices);
//     std::vector<int64_t> h_offsets(batch_size + 1);
//     std::vector<float> h_per_sample_weights(num_indices);
//     std::vector<float> h_output(batch_size * embedding_dim);
//     std::vector<float> h_output_cpu(batch_size * embedding_dim); // 用于CPU验证

//     // 3. 初始化测试数据
//     // 权重矩阵
//     for (int i = 0; i < num_embeddings * embedding_dim; i++) {
//         h_weight[i] = static_cast<float>(rand()) / RAND_MAX;
//     }
    
//     // 索引和偏移
//     h_offsets[0] = 0;
//     for (int b = 0; b < batch_size; b++) {
//         for (int i = 0; i < indices_per_batch; i++) {
//             h_indices[b * indices_per_batch + i] = rand() % num_embeddings;
//         }
//         h_offsets[b + 1] = h_offsets[b] + indices_per_batch;
//     }
    
//     // 样本权重
//     for (int i = 0; i < num_indices; i++) {
//         h_per_sample_weights[i] = static_cast<float>(rand()) / RAND_MAX;
//     }

//     // 4. 分配设备内存
//     float *d_weight, *d_output, *d_per_sample_weights;
//     int64_t *d_indices, *d_offsets;
    
//     cudaMalloc(&d_weight, num_embeddings * embedding_dim * sizeof(float));
//     cudaMalloc(&d_indices, num_indices * sizeof(int64_t));
//     cudaMalloc(&d_offsets, (batch_size + 1) * sizeof(int64_t));
//     cudaMalloc(&d_per_sample_weights, num_indices * sizeof(float));
//     cudaMalloc(&d_output, batch_size * embedding_dim * sizeof(float));

//     // 5. 复制数据到设备
//     cudaMemcpy(d_weight, h_weight.data(), num_embeddings * embedding_dim * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_indices, h_indices.data(), num_indices * sizeof(int64_t), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_offsets, h_offsets.data(), (batch_size + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_per_sample_weights, h_per_sample_weights.data(), num_indices * sizeof(float), cudaMemcpyHostToDevice);

//     // 6. 启动内核
//     int threadsPerBlock = 256;
//     int blocksPerGrid = batch_size;
    
//     // 创建CUDA事件用于计时
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
    
//     // 记录开始时间
//     cudaEventRecord(start);
    
//     embeddingbag_forward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
//         d_weight,
//         d_indices,
//         d_offsets,
//         d_per_sample_weights,
//         d_output,
//         batch_size,
//         embedding_dim,
//         true  // 使用样本权重
//     );
    
//     // 记录结束时间
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
    
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
    
//     // 7. 复制结果回主机
//     cudaMemcpy(h_output.data(), d_output, batch_size * embedding_dim * sizeof(float), cudaMemcpyDeviceToHost);

//     // 8. CPU计算参考结果
//     for (int b = 0; b < batch_size; b++) {
//         for (int d = 0; d < embedding_dim; d++) {
//             float sum = 0.0f;
//             for (int64_t idx = h_offsets[b]; idx < h_offsets[b + 1]; idx++) {
//                 int64_t embedding_idx = h_indices[idx];
//                 float w = h_per_sample_weights[idx];
//                 sum += h_weight[embedding_idx * embedding_dim + d] * w;
//             }
//             h_output_cpu[b * embedding_dim + d] = sum;
//         }
//     }

//     // 9. 验证结果
//     float max_error = 0.0f;
//     for (int i = 0; i < batch_size * embedding_dim; i++) {
//         float error = std::abs(h_output[i] - h_output_cpu[i]);
//         max_error = std::max(max_error, error);
//     }

//     // 10. 输出结果
//     printf("Embedding Bag Forward Test:\n");
//     printf("Kernel execution time: %f ms\n", milliseconds);
//     printf("Maximum error: %e\n", max_error);
//     printf("Test %s\n", max_error < 1e-5 ? "PASSED" : "FAILED");

//     // 11. 释放内存
//     cudaFree(d_weight);
//     cudaFree(d_indices);
//     cudaFree(d_offsets);
//     cudaFree(d_per_sample_weights);
//     cudaFree(d_output);
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);
// }

// int main(){
//     test_embeddingbag_forward();
//     return 0;
// }
// 保持与dlrm.cpp一致的main函数，去掉设备映射相关代码
int main(int argc, char* argv[]) {
    // 模型参数 - 与dlrm.cpp完全一致
    std::cout<<"----------------------start--------------------------------"<<std::endl;
    std::vector<int64_t> ln_emb = {int64_t(30), int64_t(20), int64_t(10)};
    std::vector<int64_t> ln_bot = {int64_t(40), int64_t(30), int64_t(50)};
    int64_t num_fea = ln_emb.size() + 1;
    int64_t m_den_out = ln_bot[ln_bot.size() - 1]; 
    int64_t num_int = (num_fea * (num_fea - 1)) / 2 + m_den_out;
    std::vector<int64_t> ln_top = {num_int, 4, 2, 1};
    int64_t mini_batch_size = 10;
    int64_t num_indices_per_lookup = 2;
    bool fixed_indices = false;
    std::string arch_interaction_op = "dot";
    bool arch_interaction_itself = false;
    int64_t sigmoid_bot = -1;
    int64_t sigmoid_top = ln_top.size() - 2;
    bool sync_dense_params = true;
    float loss_threshold = 0.0;
    
    bool qr_flag = false;
    std::string qr_operation = "mult";
    int64_t qr_collisions = 4;
    int64_t qr_threshold = 200;
    bool md_flag = false;
    int64_t md_threshold = 200;
    std::string weighted_pooling = "";
    std::string loss_function = "mse";
    
    // 使用2个GPU
    int64_t ndevices = 2;
    
    // 检查GPU可用性
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA不可用" << std::endl;
        return 1;
    }
    
    int gpu_count = torch::cuda::device_count();
    std::cout << "可用GPU数量: " << gpu_count << std::endl;
    
    if (gpu_count < ndevices) {
        std::cerr << "需要至少" << ndevices << "个GPU" << std::endl;
        return 1;
    }
    
    // 数据生成 - 与dlrm.cpp完全一致
    torch::Tensor dense_x = torch::rand({mini_batch_size, ln_bot[0]}, torch::kFloat32);
    
    if (torch::cuda::is_available() && ndevices) {
        dense_x = dense_x.to(torch::kCUDA);
        std::cout << "Using CUDA" << std::endl;
    }
    
    // ... 其他数据生成代码与dlrm.cpp保持一致 ...
    std::vector<int64_t> offsets_data;
    std::vector<std::vector<int64_t>> indices_data(ln_emb.size());
    
    std::vector<std::vector<int64_t>> indices_per_sample(ln_emb.size(), std::vector<int64_t>(mini_batch_size));
    
    for (size_t f = 0; f < ln_emb.size(); f++) {
        for (int64_t i = 0; i < mini_batch_size; i++) {
            if (fixed_indices) {
                indices_per_sample[f][i] = num_indices_per_lookup;
            } else {
                indices_per_sample[f][i] = torch::randint(1, num_indices_per_lookup + 1, {1}).item<int64_t>();
            }
        }
    }
    
    std::vector<std::vector<int64_t>> offsets_per_feature(ln_emb.size(), std::vector<int64_t>(mini_batch_size + 1, 0));
    
    for (size_t f = 0; f < ln_emb.size(); f++) {
        for (int64_t i = 0; i < mini_batch_size; i++) {
            offsets_per_feature[f][i + 1] = offsets_per_feature[f][i] + indices_per_sample[f][i];
        }
        
        indices_data[f].resize(offsets_per_feature[f][mini_batch_size]);
        for (int64_t i = 0; i < offsets_per_feature[f][mini_batch_size]; i++) {
            indices_data[f][i] = torch::randint(0, ln_emb[f], {1}).item<int64_t>();
        }
    }
    
    torch::Tensor lS_o = torch::zeros({static_cast<int64_t>(ln_emb.size()), mini_batch_size + 1}, torch::kInt64);
    for (size_t f = 0; f < ln_emb.size(); f++) {
        for (int64_t i = 0; i <= mini_batch_size; i++) {
            lS_o[f][i] = offsets_per_feature[f][i];
        }
    }
    
    std::vector<torch::Tensor> lS_i;
    for (size_t f = 0; f < ln_emb.size(); f++) {
        torch::Tensor indices_tensor = torch::zeros({static_cast<int64_t>(indices_data[f].size())}, torch::kInt64);
        for (size_t i = 0; i < indices_data[f].size(); i++) {
            indices_tensor[i] = indices_data[f][i];
        }
        lS_i.push_back(indices_tensor);
    }
    
    lS_o = lS_o.index({"...", torch::indexing::Slice(0, mini_batch_size)});
    
    // 创建模型和执行
    auto dlrm_model = DLRM(m_den_out, ln_emb, ln_bot, ln_top, arch_interaction_op, 
                          arch_interaction_itself, sigmoid_bot, sigmoid_top, sync_dense_params,
                          loss_threshold, ndevices, qr_flag, qr_operation, qr_collisions, 
                          qr_threshold, md_flag, md_threshold, weighted_pooling, loss_function);
                          
    torch::Tensor result = dlrm_model->forward(dense_x, lS_o, lS_i);
    std::cout<<"result shape: "<<result.sizes()<<std::endl;
    return 0;
}
