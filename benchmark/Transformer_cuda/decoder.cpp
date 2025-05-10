#include "apis_c.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <fstream>
#include <chrono>

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

// 生成输入数据
float* generateInputData(const DecoderConfig& config) {
    size_t input_size = config.batch_size * config.seq_len * config.hidden_size;
    float* input_data = new float[input_size];
    
    // 初始化随机输入数据
    initializeRandomData(input_data, input_size, -0.5f, 0.5f);
    
    return input_data;
}

// 主函数
int main(int argc, char** argv) {
    int srcX = 0, srcY = 0;
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
    
    std::cout << "Transformer解码器层配置:\n";
    std::cout << "批次大小: " << config.batch_size << "\n";
    std::cout << "序列长度: " << config.seq_len << "\n";
    std::cout << "隐藏层大小: " << config.hidden_size << "\n";
    std::cout << "前馈网络大小: " << config.ffn_size << "\n";
    std::cout << "注意力头数量: " << config.num_heads << "\n";
    
    // 确保头大小是隐藏层大小的整数倍
    if (config.hidden_size % config.num_heads != 0) {
        std::cerr << "错误: 隐藏层大小必须是注意力头数量的整数倍\n";
        return 1;
    }
    
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
    
    // 计算各项数据大小
    size_t input_size_bytes = output_size * sizeof(float);
    size_t attn_weights_size = 4 * config.hidden_size * config.hidden_size * sizeof(float);
    size_t attn_bias_size = 4 * config.hidden_size * sizeof(float);
    size_t ffn1_weights_size = config.hidden_size * config.ffn_size * sizeof(float);
    size_t ffn1_bias_size = config.ffn_size * sizeof(float);
    size_t ffn2_weights_size = config.ffn_size * config.hidden_size * sizeof(float);
    size_t ffn2_bias_size = config.hidden_size * sizeof(float);
    size_t ln_params_size = config.hidden_size * sizeof(float);
    
    // 将配置参数打包为数组
    int config_array[5] = {
        config.batch_size,
        config.seq_len,
        config.hidden_size,
        config.ffn_size,
        config.num_heads
    };
    
    // 目标chiplet坐标 (假设为(1,1))
    int destX = 1;
    int destY = 0;
    
    std::cout << "开始向CUDA设备发送数据...\n";
    
    // 1. 发送配置参数
    InterChiplet::sendMessage( destX, destY, srcX, srcY,config_array, sizeof(int) * 5);
    std::cout << "已发送配置参数\n";
    
    // 2. 发送模型参数
    InterChiplet::sendMessage( destX, destY,srcX, srcY, attn_weights, attn_weights_size);
    InterChiplet::sendMessage( destX, destY,srcX, srcY, attn_bias, attn_bias_size);
    InterChiplet::sendMessage( destX, destY,srcX, srcY, ffn_weights1, ffn1_weights_size);
    InterChiplet::sendMessage( destX, destY,srcX, srcY, ffn_bias1, ffn1_bias_size);
    InterChiplet::sendMessage( destX, destY,srcX, srcY, ffn_weights2, ffn2_weights_size);
    InterChiplet::sendMessage( destX, destY,srcX, srcY, ffn_bias2, ffn2_bias_size);
    InterChiplet::sendMessage( destX, destY,srcX, srcY, ln1_gamma, ln_params_size);
    InterChiplet::sendMessage( destX, destY,srcX, srcY, ln1_beta, ln_params_size);
    InterChiplet::sendMessage( destX, destY,srcX, srcY, ln2_gamma, ln_params_size);
    InterChiplet::sendMessage( destX, destY,srcX, srcY, ln2_beta, ln_params_size);
    std::cout << "已发送所有模型参数\n";
    
    // 3. 发送输入数据
    InterChiplet::sendMessage( destX, destY,srcX, srcY, input_data, input_size_bytes);
    
    // 4. 接收输出数据
    InterChiplet::receiveMessage(srcX, srcY, destX, destY, output_data, input_size_bytes);
    
    // 输出部分结果用于验证
    std::cout << "\n输出样本 (前10个值):\n";
    for (int i = 0; i < 10; i++) {
        std::cout << output_data[i] << " ";
    }
    std::cout << "\n";
    
    // 将结果保存到文件
    // std::ofstream result_file("decoder_output.bin", std::ios::binary);
    // if (result_file.is_open()) {
    //     result_file.write(reinterpret_cast<char*>(output_data), output_size * sizeof(float));
    //     result_file.close();
    //     std::cout << "结果已保存到decoder_output.bin\n";
    // } else {
    //     std::cerr << "无法创建输出文件\n";
    // }
    
    // 释放内存
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
    delete[] input_data;
    delete[] output_data;
    
    std::cout << "程序执行完毕\n";
    return 0;
}