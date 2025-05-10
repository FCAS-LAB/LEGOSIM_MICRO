#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <string>
#include <iostream>
// #include "apis_cu.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <random>

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
    
    // 第三步：多头自注意力
    dim3 attn_block(BLOCK_SIZE);
    dim3 attn_grid(seq_len, num_heads, batch_size);
    int shared_mem_size = BLOCK_SIZE * seq_len * sizeof(float);
    
    self_attention_kernel<<<attn_grid, attn_block, shared_mem_size>>>(
        q_out, k_out, v_out, attn_out, 
        batch_size, seq_len, head_size, num_heads
    );
    cudaDeviceSynchronize();
    
    // 第四步：注意力输出投影
    matmul_kernel<<<qkv_grid, mm_block>>>(
        attn_out, attn_weights + 3 * hidden_size * hidden_size, 
        attn_dropout, batch_size * seq_len, hidden_size, hidden_size
    );
    cudaDeviceSynchronize();
    
    // 第五步：第一个残差连接
    dim3 add_grid((batch_size * seq_len * hidden_size + 255) / 256);
    add_residual_kernel<<<add_grid, 256>>>(input, attn_dropout, add1_out, batch_size * seq_len * hidden_size);
    cudaDeviceSynchronize();
    
    // 第六步：第二个层归一化
    layer_norm_kernel<<<ln_grid, ln_block>>>(add1_out, norm2_out, ln2_gamma, ln2_beta, hidden_size);
    cudaDeviceSynchronize();
    
    // 第七步：前馈网络第一层
    dim3 ffn1_grid((ffn_size + mm_block.x - 1) / mm_block.x,
                   (batch_size * seq_len + mm_block.y - 1) / mm_block.y);
    
    matmul_kernel<<<ffn1_grid, mm_block>>>(
        norm2_out, ffn_weights1, 
        ffn1_out, batch_size * seq_len, ffn_size, hidden_size
    );
    cudaDeviceSynchronize();
    // 添加偏置并应用GELU激活
    dim3 gelu_grid((batch_size * seq_len * ffn_size + 255) / 256);
    gelu_kernel<<<gelu_grid, 256>>>(ffn1_out, gelu_out, batch_size * seq_len * ffn_size);
    cudaDeviceSynchronize();
    
    // 第八步：前馈网络第二层
    dim3 ffn2_grid((hidden_size + mm_block.x - 1) / mm_block.x,
                   (batch_size * seq_len + mm_block.y - 1) / mm_block.y);
    
    matmul_kernel<<<ffn2_grid, mm_block>>>(
        gelu_out, ffn_weights2, 
        ffn2_out, batch_size * seq_len, hidden_size, ffn_size
    );
    cudaDeviceSynchronize();
    // 第九步：第二个残差连接
    add_residual_kernel<<<add_grid, 256>>>(add1_out, ffn2_out, output, batch_size * seq_len * hidden_size);
    cudaDeviceSynchronize();
}

// Transformer解码器层配置参数
struct DecoderConfig {
    int batch_size;
    int seq_len;
    int hidden_size;
    int ffn_size;
    int num_heads;
};

// 初始化随机数据
void initializeRandomData(float* data, size_t size, float min = -0.1f, float max = 0.1f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    
    for (size_t i = 0; i < size; i++) {
        data[i] = dist(gen);
    }
}

// 初始化模型参数
void initializeModelParameters(
    const DecoderConfig& config,
    float*& attn_weights, float*& attn_bias,
    float*& ffn_weights1, float*& ffn_bias1,
    float*& ffn_weights2, float*& ffn_bias2,
    float*& ln1_gamma, float*& ln1_beta,
    float*& ln2_gamma, float*& ln2_beta
) {
    int hidden_size = config.hidden_size;
    int ffn_size = config.ffn_size;
    
    // 分配内存
    size_t attn_weights_size = 4 * hidden_size * hidden_size;
    size_t attn_bias_size = 4 * hidden_size;
    size_t ffn1_weights_size = hidden_size * ffn_size;
    size_t ffn1_bias_size = ffn_size;
    size_t ffn2_weights_size = ffn_size * hidden_size;
    size_t ffn2_bias_size = hidden_size;
    size_t ln_params_size = hidden_size;
    
    attn_weights = new float[attn_weights_size];
    attn_bias = new float[attn_bias_size];
    ffn_weights1 = new float[ffn1_weights_size];
    ffn_bias1 = new float[ffn1_bias_size];
    ffn_weights2 = new float[ffn2_weights_size];
    ffn_bias2 = new float[ffn2_bias_size];
    ln1_gamma = new float[ln_params_size];
    ln1_beta = new float[ln_params_size];
    ln2_gamma = new float[ln_params_size];
    ln2_beta = new float[ln_params_size];
    
    // 初始化注意力权重和偏置
    initializeRandomData(attn_weights, attn_weights_size, -0.02f, 0.02f);
    initializeRandomData(attn_bias, attn_bias_size, -0.02f, 0.02f);
    
    // 初始化前馈网络权重和偏置
    initializeRandomData(ffn_weights1, ffn1_weights_size, -0.02f, 0.02f);
    initializeRandomData(ffn_bias1, ffn1_bias_size, -0.02f, 0.02f);
    initializeRandomData(ffn_weights2, ffn2_weights_size, -0.02f, 0.02f);
    initializeRandomData(ffn_bias2, ffn2_bias_size, -0.02f, 0.02f);
    
    // 初始化层归一化参数
    for (int i = 0; i < int(ln_params_size); i++) {
        ln1_gamma[i] = 1.0f;  // 默认为1
        ln1_beta[i] = 0.0f;   // 默认为0
        ln2_gamma[i] = 1.0f;  // 默认为1
        ln2_beta[i] = 0.0f;   // 默认为0
    }
}
float* generateInputData(const DecoderConfig& config) {
    size_t input_size = config.batch_size * config.seq_len * config.hidden_size;
    float* input_data = new float[input_size];
    
    // 初始化随机输入数据
    initializeRandomData(input_data, input_size, -0.5f, 0.5f);
    
    return input_data;
}

int main() {
    // 计时变量
    struct timeval start_cpu, end_cpu;
    struct timeval start_transfer_h2d, end_transfer_h2d;
    struct timeval start_compute, end_compute;
    struct timeval start_transfer_d2h, end_transfer_d2h;
    float cpu_time, transfer_h2d_time, compute_time, transfer_d2h_time;
    
    // 开始CPU计时
    
    
    // 分配设备内存
    float *d_input, *d_output, *d_buffer;
    float *d_attn_weights, *d_attn_bias;
    float *d_ffn_weights1, *d_ffn_bias1;
    float *d_ffn_weights2, *d_ffn_bias2;
    float *d_ln1_gamma, *d_ln1_beta;
    float *d_ln2_gamma, *d_ln2_beta;

    // 设置解码器层配置参数
    DecoderConfig config;
    config.batch_size = 1;       // 批次大小
    config.seq_len = 128;        // 序列长度
    config.hidden_size = 64;    // 隐藏层大小
    config.ffn_size = config.hidden_size*4;      // 前馈网络中间层大小
    config.num_heads = config.hidden_size/16;       // 注意力头数量
    
    // 如果命令行提供了参数，则覆盖默认配置
    // if (argc > 1) config.batch_size = std::atoi(argv[1]);
    // if (argc > 2) config.seq_len = std::atoi(argv[2]);
    // if (argc > 3) config.hidden_size = std::atoi(argv[3]);
    // if (argc > 4) config.ffn_size = std::atoi(argv[4]);
    // if (argc > 5) config.num_heads = std::atoi(argv[5]);
    
    // std::cout << "Transformer解码器层配置:\n";
    // std::cout << "批次大小: " << config.batch_size << "\n";
    // std::cout << "序列长度: " << config.seq_len << "\n";
    // std::cout << "隐藏层大小: " << config.hidden_size << "\n";
    // std::cout << "前馈网络大小: " << config.ffn_size << "\n";
    // std::cout << "注意力头数量: " << config.num_heads << "\n";
    
    // 确保头大小是隐藏层大小的整数倍
    if (config.hidden_size % config.num_heads != 0) {
        std::cerr << "错误: 隐藏层大小必须是注意力头数量的整数倍\n";
        return 1;
    }
    int batch_size = config.batch_size;
    int seq_len = config.seq_len;
    int hidden_size = config.hidden_size;
    int ffn_size = config.ffn_size;
    int num_heads = config.num_heads;
    
    
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
    gettimeofday(&start_cpu, NULL);
    // 初始化模型参数
    float *attn_weights, *attn_bias;
    float *ffn_weights1, *ffn_bias1;
    float *ffn_weights2, *ffn_bias2;
    float *ln1_gamma, *ln1_beta;
    float *ln2_gamma, *ln2_beta;
    
    initializeModelParameters(
        config,
        attn_weights, attn_bias,
        ffn_weights1, ffn_bias1,
        ffn_weights2, ffn_bias2,
        ln1_gamma, ln1_beta,
        ln2_gamma, ln2_beta
    );
    
    // 生成输入数据
    float* input_data = generateInputData(config);
    
    // 为输出数据分配内存
    size_t output_size = config.batch_size * config.seq_len * config.hidden_size;
    float* output_data = new float[output_size];
    
    // 开始Host到Device数据传输计时
    gettimeofday(&start_transfer_h2d, NULL);
    
    cudaMemcpy(d_input, input_data, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_attn_weights, attn_weights, attn_weights_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_attn_bias, attn_bias, attn_bias_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ffn_weights1, ffn_weights1, ffn1_weights_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ffn_bias1, ffn_bias1, ffn1_bias_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ffn_weights2, ffn_weights2, ffn2_weights_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ffn_bias2, ffn_bias2, ffn2_bias_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ln1_gamma, ln1_gamma, ln_params_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ln1_beta, ln1_beta, ln_params_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ln2_gamma, ln2_gamma, ln_params_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ln2_beta, ln2_beta, ln_params_size, cudaMemcpyHostToDevice);
    
    // 结束Host到Device数据传输计时
    gettimeofday(&end_transfer_h2d, NULL);
    
    printf("接收到配置: batch_size=%d, seq_len=%d, hidden_size=%d, ffn_size=%d, num_heads=%d\n",
            batch_size, seq_len, hidden_size, ffn_size, num_heads);
    
    // 打印接收到的输入序列信息
    printf("接收到输入序列，大小: %lu 字节\n", input_size);
    
    // 开始GPU计算计时
    gettimeofday(&start_compute, NULL);
    
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
    
    // 结束GPU计算计时
    gettimeofday(&end_compute, NULL);
    
    // 开始Device到Host数据传输计时
    gettimeofday(&start_transfer_d2h, NULL);
    
    // 将结果发送回CPU
    cudaMemcpy(output_data, d_output, input_size, cudaMemcpyDeviceToHost);
    
    // 结束Device到Host数据传输计时
    gettimeofday(&end_transfer_d2h, NULL);
    
    // 结束CPU计时
    gettimeofday(&end_cpu, NULL);
    
    // 计算各部分耗时（以毫秒为单位）
    cpu_time = (end_cpu.tv_sec - start_cpu.tv_sec) * 1000.0 + 
               (end_cpu.tv_usec - start_cpu.tv_usec) / 1000.0;
               
    transfer_h2d_time = (end_transfer_h2d.tv_sec - start_transfer_h2d.tv_sec) * 1000.0 + 
                        (end_transfer_h2d.tv_usec - start_transfer_h2d.tv_usec) / 1000.0;
                        
    compute_time = (end_compute.tv_sec - start_compute.tv_sec) * 1000.0 + 
                  (end_compute.tv_usec - start_compute.tv_usec) / 1000.0;
                  
    transfer_d2h_time = (end_transfer_d2h.tv_sec - start_transfer_d2h.tv_sec) * 1000.0 + 
                        (end_transfer_d2h.tv_usec - start_transfer_d2h.tv_usec) / 1000.0;
    
    // 打印时间统计信息
    printf("\n========== 时间统计 ==========\n");
    printf("CPU总耗时: %.3f 毫秒\n", cpu_time);
    printf("Host->Device数据传输时间: %.3f 毫秒\n", transfer_h2d_time);
    printf("GPU计算时间: %.3f 毫秒\n", compute_time);
    printf("Device->Host数据传输时间: %.3f 毫秒\n", transfer_d2h_time);
    printf("CPU开销 (不包括传输和计算): %.3f 毫秒\n", 
           cpu_time - (transfer_h2d_time + compute_time + transfer_d2h_time));
    printf("==============================\n");

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
    
    delete[] input_data;
    delete[] output_data;
    delete[] attn_weights;
    delete[] attn_bias;
    delete[] ffn_weights1;
    delete[] ffn_bias1;
    delete[] ffn_weights2;
    delete[] ffn_bias2;
    delete[] ln1_gamma;
    delete[] ln1_beta;
    delete[] ln2_gamma;
    delete[] ln2_beta;
    
    return 0;
}