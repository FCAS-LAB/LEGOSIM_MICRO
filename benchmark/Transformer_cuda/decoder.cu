#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <string>
#include <iostream>
#include "apis_cu.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 16
#define WARP_SIZE 32
#define MASK_VAL -10000.0f

// 矩阵乘法核函数
__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Layer Normalization核函数
__global__ void layer_norm_kernel(float* input, float* output, float* gamma, float* beta, int hidden_size) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int offset = (batch_idx * gridDim.y + seq_idx) * hidden_size;
    
    // 共享内存存储部分和
    __shared__ float mean_shared, var_shared;
    
    // 计算均值的第一步 - 求和
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        local_sum += input[offset + i];
    }
    
    // 归约求和
    __shared__ float sum_shared[256];
    sum_shared[threadIdx.x] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum_shared[threadIdx.x] += sum_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // 计算均值
    if (threadIdx.x == 0) {
        mean_shared = sum_shared[0] / hidden_size;
    }
    __syncthreads();
    
    // 计算方差的第一步 - 求平方差之和
    local_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = input[offset + i] - mean_shared;
        local_sum += diff * diff;
    }
    
    // 归约求和
    sum_shared[threadIdx.x] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum_shared[threadIdx.x] += sum_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // 计算方差
    if (threadIdx.x == 0) {
        var_shared = sum_shared[0] / hidden_size;
    }
    __syncthreads();
    
    // 应用层归一化
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (input[offset + i] - mean_shared) / sqrtf(var_shared + 1e-5f);
        output[offset + i] = gamma[i] * normalized + beta[i];
    }
}

// GELU激活函数
__global__ void gelu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = input[idx];
        // GELU近似: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        output[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

// 多头自注意力核函数
__global__ void self_attention_kernel(float* query, float* key, float* value, float* output,
                                     int batch_size, int seq_len, int head_size, int num_heads) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= seq_len) return;
    
    extern __shared__ float shared_mem[];
    float* scores = shared_mem;
    
    // 计算注意力分数
    for (int col = 0; col < seq_len; col++) {
        float dot_product = 0.0f;
        
        for (int i = 0; i < head_size; i++) {
            int q_idx = ((b * seq_len + row) * num_heads + h) * head_size + i;
            int k_idx = ((b * seq_len + col) * num_heads + h) * head_size + i;
            dot_product += query[q_idx] * key[k_idx];
        }
        
        scores[threadIdx.x * seq_len + col] = dot_product / sqrtf((float)head_size);
        
        // 因果掩码（只关注自己和之前的位置）
        if (col > row) {
            scores[threadIdx.x * seq_len + col] = MASK_VAL;
        }
    }
    
    // Softmax
    float max_val = MASK_VAL;
    for (int i = 0; i < seq_len; i++) {
        max_val = fmaxf(max_val, scores[threadIdx.x * seq_len + i]);
    }
    
    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        scores[threadIdx.x * seq_len + i] = expf(scores[threadIdx.x * seq_len + i] - max_val);
        sum += scores[threadIdx.x * seq_len + i];
    }
    
    for (int i = 0; i < seq_len; i++) {
        scores[threadIdx.x * seq_len + i] /= sum;
    }
    
    // 加权求和
    for (int i = 0; i < head_size; i++) {
        float weighted_sum = 0.0f;
        
        for (int j = 0; j < seq_len; j++) {
            int v_idx = ((b * seq_len + j) * num_heads + h) * head_size + i;
            weighted_sum += scores[threadIdx.x * seq_len + j] * value[v_idx];
        }
        
        int out_idx = ((b * seq_len + row) * num_heads + h) * head_size + i;
        output[out_idx] = weighted_sum;
    }
}

// 添加一个残差连接kernel
__global__ void add_residual_kernel(float* input, float* residual, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] + residual[idx];
    }
}

// 拆分解码器层的功能为多个独立kernel函数
void run_decoder_layer(
    float* input,         // 输入张量 [batch_size, seq_len, hidden_size]
    float* attn_weights,  // 注意力权重参数
    float* attn_bias,     // 注意力偏置
    float* ffn_weights1,  // 前馈网络第一层权重
    float* ffn_bias1,     // 前馈网络第一层偏置
    float* ffn_weights2,  // 前馈网络第二层权重
    float* ffn_bias2,     // 前馈网络第二层偏置
    float* ln1_gamma,     // 第一个层归一化gamma
    float* ln1_beta,      // 第一个层归一化beta
    float* ln2_gamma,     // 第二个层归一化gamma
    float* ln2_beta,      // 第二个层归一化beta
    float* buffer,        // 计算缓冲区
    float* output,        // 输出张量 [batch_size, seq_len, hidden_size]
    int batch_size,       // 批次大小
    int seq_len,          // 序列长度
    int hidden_size,      // 隐藏层大小
    int ffn_size,         // 前馈网络中间层大小
    int num_heads         // 注意力头数量
) {
    // 计算头大小
    int head_size = hidden_size / num_heads;
    
    // 缓冲区分配
    float* norm1_out = buffer;
    float* q_out = norm1_out + batch_size * seq_len * hidden_size;
    float* k_out = q_out + batch_size * seq_len * hidden_size;
    float* v_out = k_out + batch_size * seq_len * hidden_size;
    float* attn_out = v_out + batch_size * seq_len * hidden_size;
    float* attn_dropout = attn_out + batch_size * seq_len * hidden_size;
    float* add1_out = attn_dropout + batch_size * seq_len * hidden_size;
    float* norm2_out = add1_out + batch_size * seq_len * hidden_size;
    float* ffn1_out = norm2_out + batch_size * seq_len * hidden_size;
    float* gelu_out = ffn1_out + batch_size * seq_len * ffn_size;
    float* ffn2_out = gelu_out + batch_size * seq_len * ffn_size;
    
    // 第一步：层归一化
    dim3 ln_block(256);
    dim3 ln_grid(batch_size, seq_len);
    layer_norm_kernel<<<ln_grid, ln_block>>>(input, norm1_out, ln1_gamma, ln1_beta, hidden_size);
    cudaDeviceSynchronize();
    
    std::cout<<"#############################################step1#############################################"<<std::endl;
    std::cout<<"norm1_out: "<<norm1_out<<std::endl;
    std::cout<<"ln1_gamma: "<<ln1_gamma<<std::endl;
    std::cout<<"ln1_beta: "<<ln1_beta<<std::endl;
    std::cout<<"hidden_size: "<<hidden_size<<std::endl;
    std::cout<<"#############################################step1#############################################"<<std::endl;
    // 第二步：QKV投影
    dim3 mm_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 qkv_grid((hidden_size + mm_block.x - 1) / mm_block.x, 
                 (batch_size * seq_len + mm_block.y - 1) / mm_block.y);
    
    // Q投影
    matmul_kernel<<<qkv_grid, mm_block>>>(
        norm1_out, attn_weights, 
        q_out, batch_size * seq_len, hidden_size, hidden_size
    );
    cudaDeviceSynchronize();
    
    // K投影
    matmul_kernel<<<qkv_grid, mm_block>>>(
        norm1_out, attn_weights + hidden_size * hidden_size, 
        k_out, batch_size * seq_len, hidden_size, hidden_size
    );
    cudaDeviceSynchronize();
    
    // V投影
    matmul_kernel<<<qkv_grid, mm_block>>>(
        norm1_out, attn_weights + 2 * hidden_size * hidden_size, 
        v_out, batch_size * seq_len, hidden_size, hidden_size
    );
    cudaDeviceSynchronize();
    
    std::cout<<"#############################################step2#############################################"<<std::endl;
    std::cout<<"q_out: "<<q_out<<std::endl;
    std::cout<<"k_out: "<<k_out<<std::endl;
    std::cout<<"v_out: "<<v_out<<std::endl;
    std::cout<<"#############################################step2#############################################"<<std::endl;
    // 第三步：多头自注意力
    dim3 attn_block(BLOCK_SIZE);
    dim3 attn_grid(seq_len, num_heads, batch_size);
    int shared_mem_size = BLOCK_SIZE * seq_len * sizeof(float);
    
    self_attention_kernel<<<attn_grid, attn_block, shared_mem_size>>>(
        q_out, k_out, v_out, attn_out, 
        batch_size, seq_len, head_size, num_heads
    );
    cudaDeviceSynchronize();
    std::cout<<"#############################################step3#############################################"<<std::endl;
    std::cout<<"attn_out: "<<attn_out<<std::endl;
    std::cout<<"#############################################step3#############################################"<<std::endl;
    
    // 第四步：注意力输出投影
    matmul_kernel<<<qkv_grid, mm_block>>>(
        attn_out, attn_weights + 3 * hidden_size * hidden_size, 
        attn_dropout, batch_size * seq_len, hidden_size, hidden_size
    );
    cudaDeviceSynchronize();
    std::cout<<"#############################################step4#############################################"<<std::endl;
    std::cout<<"attn_dropout: "<<attn_dropout<<std::endl;
    std::cout<<"#############################################step4#############################################"<<std::endl;
    
    // 第五步：第一个残差连接
    dim3 add_grid((batch_size * seq_len * hidden_size + 255) / 256);
    add_residual_kernel<<<add_grid, 256>>>(input, attn_dropout, add1_out, batch_size * seq_len * hidden_size);
    cudaDeviceSynchronize();
    std::cout<<"#############################################step5#############################################"<<std::endl;
    std::cout<<"add1_out: "<<add1_out<<std::endl;
    std::cout<<"#############################################step5#############################################"<<std::endl;
    
    // 第六步：第二个层归一化
    layer_norm_kernel<<<ln_grid, ln_block>>>(add1_out, norm2_out, ln2_gamma, ln2_beta, hidden_size);
    cudaDeviceSynchronize();
    std::cout<<"#############################################step6#############################################"<<std::endl;
    std::cout<<"norm2_out: "<<norm2_out<<std::endl;
    std::cout<<"#############################################step6#############################################"<<std::endl;
    
    // 第七步：前馈网络第一层
    dim3 ffn1_grid((ffn_size + mm_block.x - 1) / mm_block.x,
                   (batch_size * seq_len + mm_block.y - 1) / mm_block.y);
    
    matmul_kernel<<<ffn1_grid, mm_block>>>(
        norm2_out, ffn_weights1, 
        ffn1_out, batch_size * seq_len, ffn_size, hidden_size
    );
    cudaDeviceSynchronize();
    std::cout<<"#############################################step7#############################################"<<std::endl;
    std::cout<<"ffn1_out: "<<ffn1_out<<std::endl;
    std::cout<<"#############################################step7#############################################"<<std::endl;
    // 添加偏置并应用GELU激活
    dim3 gelu_grid((batch_size * seq_len * ffn_size + 255) / 256);
    gelu_kernel<<<gelu_grid, 256>>>(ffn1_out, gelu_out, batch_size * seq_len * ffn_size);
    cudaDeviceSynchronize();
    std::cout<<"#############################################step8#############################################"<<std::endl;
    std::cout<<"gelu_out: "<<gelu_out<<std::endl;
    std::cout<<"#############################################step8#############################################"<<std::endl;
    
    // 第八步：前馈网络第二层
    dim3 ffn2_grid((hidden_size + mm_block.x - 1) / mm_block.x,
                   (batch_size * seq_len + mm_block.y - 1) / mm_block.y);
    
    matmul_kernel<<<ffn2_grid, mm_block>>>(
        gelu_out, ffn_weights2, 
        ffn2_out, batch_size * seq_len, hidden_size, ffn_size
    );
    cudaDeviceSynchronize();
    std::cout<<"#############################################step9#############################################"<<std::endl;
    std::cout<<"ffn2_out: "<<ffn2_out<<std::endl;
    std::cout<<"#############################################step9#############################################"<<std::endl;
    // 第九步：第二个残差连接
    add_residual_kernel<<<add_grid, 256>>>(add1_out, ffn2_out, output, batch_size * seq_len * hidden_size);
    cudaDeviceSynchronize();
    std::cout<<"#############################################step10#############################################"<<std::endl;
    std::cout<<"output: "<<output<<std::endl;
    std::cout<<"#############################################step10#############################################"<<std::endl;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("用法: %s <srcX> <srcY>\n", argv[0]);
        return 1;
    }
    
    // 读取chiplet编号
    int srcX = atoi(argv[1]);
    int srcY = atoi(argv[2]);
    
    // 首先接收模型配置参数
    int *config = new int[5]; // batch_size, seq_len, hidden_size, ffn_size, num_heads
    int *d_config;
    cudaMalloc((void **)&d_config, sizeof(int) * 5);
    
    // 从CPU接收配置
    receiveMessage(srcX, srcY, 0, 0, d_config, sizeof(int) * 5);
    cudaMemcpy(config, d_config, sizeof(int) * 5, cudaMemcpyDeviceToHost);
    
    int batch_size = config[0];
    int seq_len = config[1];
    int hidden_size = config[2];
    int ffn_size = config[3];
    int num_heads = config[4];
    
    printf("接收到配置: batch_size=%d, seq_len=%d, hidden_size=%d, ffn_size=%d, num_heads=%d\n",
            batch_size, seq_len, hidden_size, ffn_size, num_heads);
    
    // 分配设备内存
    float *d_input, *d_output, *d_buffer;
    float *d_attn_weights, *d_attn_bias;
    float *d_ffn_weights1, *d_ffn_bias1;
    float *d_ffn_weights2, *d_ffn_bias2;
    float *d_ln1_gamma, *d_ln1_beta;
    float *d_ln2_gamma, *d_ln2_beta;
    
    size_t input_size = batch_size * seq_len * hidden_size * sizeof(float);
    size_t attn_weights_size = 4 * hidden_size * hidden_size * sizeof(float);
    size_t attn_bias_size = 4 * hidden_size * sizeof(float);
    size_t ffn1_weights_size = hidden_size * ffn_size * sizeof(float);
    size_t ffn1_bias_size = ffn_size * sizeof(float);
    size_t ffn2_weights_size = ffn_size * hidden_size * sizeof(float);
    size_t ffn2_bias_size = hidden_size * sizeof(float);
    size_t ln_params_size = hidden_size * sizeof(float);
    
    // 计算所需缓冲区大小
    size_t buffer_size = (
        batch_size * seq_len * hidden_size * 8 + // 注意力相关
        batch_size * seq_len * ffn_size * 2 +   // 前馈网络相关
        batch_size * seq_len * hidden_size * 2  // 残差相关
    ) * sizeof(float);
    
    // 分配设备内存
    cudaMalloc((void **)&d_input, input_size);
    cudaMalloc((void **)&d_output, input_size);
    cudaMalloc((void **)&d_buffer, buffer_size);
    cudaMalloc((void **)&d_attn_weights, attn_weights_size);
    cudaMalloc((void **)&d_attn_bias, attn_bias_size);
    cudaMalloc((void **)&d_ffn_weights1, ffn1_weights_size);
    cudaMalloc((void **)&d_ffn_bias1, ffn1_bias_size);
    cudaMalloc((void **)&d_ffn_weights2, ffn2_weights_size);
    cudaMalloc((void **)&d_ffn_bias2, ffn2_bias_size);
    cudaMalloc((void **)&d_ln1_gamma, ln_params_size);
    cudaMalloc((void **)&d_ln1_beta, ln_params_size);
    cudaMalloc((void **)&d_ln2_gamma, ln_params_size);
    cudaMalloc((void **)&d_ln2_beta, ln_params_size);
    
    // 接收模型参数
    receiveMessage(srcX, srcY, 0, 0, d_attn_weights, attn_weights_size);
    receiveMessage(srcX, srcY, 0, 0, d_attn_bias, attn_bias_size);
    receiveMessage(srcX, srcY, 0, 0, d_ffn_weights1, ffn1_weights_size);
    receiveMessage(srcX, srcY, 0, 0, d_ffn_bias1, ffn1_bias_size);
    receiveMessage(srcX, srcY, 0, 0, d_ffn_weights2, ffn2_weights_size);
    receiveMessage(srcX, srcY, 0, 0, d_ffn_bias2, ffn2_bias_size);
    receiveMessage(srcX, srcY, 0, 0, d_ln1_gamma, ln_params_size);
    receiveMessage(srcX, srcY, 0, 0, d_ln1_beta, ln_params_size);
    receiveMessage(srcX, srcY, 0, 0, d_ln2_gamma, ln_params_size);
    receiveMessage(srcX, srcY, 0, 0, d_ln2_beta, ln_params_size);
    
    // 接收输入序列
    receiveMessage(srcX, srcY, 0, 0, d_input, input_size);
    
    // 打印接收到的输入序列信息
    printf("接收到输入序列，大小: %lu 字节\n", input_size);
    
    // 执行Transformer解码器层计算
    run_decoder_layer(
        d_input, d_attn_weights, d_attn_bias,
        d_ffn_weights1, d_ffn_bias1, 
        d_ffn_weights2, d_ffn_bias2,
        d_ln1_gamma, d_ln1_beta,
        d_ln2_gamma, d_ln2_beta,
        d_buffer, d_output,
        batch_size, seq_len, hidden_size, ffn_size, num_heads
    );
    
    // 将结果发送回CPU
    sendMessage(0, 0, srcX, srcY, d_output, input_size);
    printf("解码器层计算完成，结果已发送回CPU\n");
    
    // 释放内存
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_buffer);
    cudaFree(d_attn_weights);
    cudaFree(d_attn_bias);
    cudaFree(d_ffn_weights1);
    cudaFree(d_ffn_bias1);
    cudaFree(d_ffn_weights2);
    cudaFree(d_ffn_bias2);
    cudaFree(d_ln1_gamma);
    cudaFree(d_ln1_beta);
    cudaFree(d_ln2_gamma);
    cudaFree(d_ln2_beta);
    cudaFree(d_config);
    
    delete[] config;

    
    return 0;
}