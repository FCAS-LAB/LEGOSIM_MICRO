#include <torch/torch.h>
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
#include "apis_c.h"
#include "cmdline_opt.h"


int max_seq_len = 1000;


/*
    矩阵分发函数
    函数名：distribute_matrix()
    输入参数：
        float* matrix: 矩阵
        vector<int64_t> dim_array: 矩阵维度数组
        int64_t dim_size: 矩阵维度大小
        int64_t matrix_size: 矩阵大小
        int src_x: 源矩阵x坐标
        int src_y: 源矩阵y坐标
        int dst_x: 目标矩阵x坐标
        int dst_y: 目标矩阵y坐标
    输出参数：
        float* matrix: 矩阵
*/

// void distribute_matrix(float* matrix, int64_t matrix_size, int src_x, int src_y, int dst_x, int dst_y){

// }


float* tensor2array(torch::Tensor tensor){
    // 确保张量是Float类型
    tensor = tensor.to(torch::kFloat32);
    auto tensor_data = tensor.data_ptr<float>();
    auto tensor_size = tensor.numel();
    float* array = new float[tensor_size];
    for(int i = 0; i < tensor_size; i++){
        array[i] = tensor_data[i];
    }
    return array;
}


torch::Tensor custom_matmul(torch::Tensor a, torch::Tensor b){
    // 确保输入张量是Float类型
    a = a.to(torch::kFloat32);
    b = b.to(torch::kFloat32);
    
    float* a_array = tensor2array(a);
    float* b_array = tensor2array(b);

    std::vector<int64_t> a_shape = a.sizes().vec();
    std::vector<int64_t> b_shape = b.sizes().vec();
    std::vector<int64_t> c_shape = a.sizes().vec();
    c_shape[c_shape.size() - 1] = b_shape[b_shape.size() - 1];

    // 分配结果数组
    int64_t c_size = 1;
    for(auto& dim : c_shape) {
        c_size *= dim;
    }
    float* c_array = new float[c_size];

    // 最后两个维度是进行矩阵乘法维度
    // a的形状为(...,m,k), b的形状为(...,k,n), c的形状为(...,m,n)
    int64_t m = a_shape[a_shape.size() - 2];
    int64_t k = a_shape[a_shape.size() - 1];
    int64_t n = b_shape[b_shape.size() - 1];

    // 计算批次维度的大小（批次维度是除最后两个维度外的所有维度）
    int64_t batch_size = 1;
    for (size_t i = 0; i < a_shape.size() - 2; ++i) {
        batch_size *= a_shape[i];
    }

    for(int64_t batch = 0; batch < batch_size; batch++){
        int64_t a_offset = batch * m * k;
        int64_t b_offset = batch * k * n;
        int64_t c_offset = batch * m * n;

        for(int64_t i = 0; i < m; i++){
            for(int64_t j = 0; j < n; j++){
                float sum = 0.0;
                for(int64_t p = 0; p < k; p++){
                    sum += a_array[a_offset + i * k + p] * b_array[b_offset + p * n + j];
                }
                c_array[c_offset + i * n + j] = sum;
            }
        }
    }
    
    delete[] a_array;
    delete[] b_array;
    
    // 创建结果张量并复制数据
    torch::Tensor c = torch::from_blob(c_array, c_shape, [c_array](void* p) { delete[] c_array; }, torch::kFloat32);
    
    return c.clone();
}

void custom_matmul_GPU(torch::Tensor a, torch::Tensor b, torch::Tensor &c, int srcX, int srcY, int dstX, int dstY){
    // 确保输入张量是Float类型
    a = a.to(torch::kFloat32);
    b = b.to(torch::kFloat32);
    
    float* a_array = tensor2array(a);
    float* b_array = tensor2array(b);
    int64_t a_size = a.numel();
    int64_t b_size = b.numel();

    std::vector<int64_t> a_shape = a.sizes().vec();
    std::vector<int64_t> b_shape = b.sizes().vec();
    std::vector<int64_t> c_shape = a.sizes().vec();
    c_shape[c_shape.size() - 1] = b_shape[b_shape.size() - 1];

    // 分配结果数组
    int64_t c_size = 1;
    for(auto& dim : c_shape) {
        c_size *= dim;
    }
    float* c_array = new float[c_size];

    // 最后两个维度是进行矩阵乘法维度
    // a的形状为(...,m,k), b的形状为(...,k,n), c的形状为(...,m,n)
    int64_t m = a_shape[a_shape.size() - 2];
    int64_t k = a_shape[a_shape.size() - 1];
    int64_t n = b_shape[b_shape.size() - 1];

    // 计算批次维度的大小（批次维度是除最后两个维度外的所有维度）
    int64_t batch_size = 1;
    for (size_t i = 0; i < a_shape.size() - 2; ++i) {
        batch_size *= a_shape[i];
    }

    /* 
        发送给GPU的数据：
            a_array: [batch_size, m, k]
            b_array: [batch_size, k, n]
        
        接收GPU的数据：
            c_array: [batch_size, m, n]
    */
    int64_t send_size[4];
    send_size[0] = batch_size;
    send_size[1] = m;
    send_size[2] = k;
    send_size[3] = n;
    bool is_end = false;
    InterChiplet::sendMessage(dstX, dstY, srcX, srcY, &is_end, sizeof(bool));
    InterChiplet::sendMessage(dstX, dstY, srcX, srcY, send_size, 4*sizeof(int64_t));
    // InterChiplet::sendMessage(dstX, dstY, srcX, srcY, &batch_size, sizeof(int64_t));
    // InterChiplet::sendMessage(dstX, dstY, srcX, srcY, &m, sizeof(int64_t));
    // InterChiplet::sendMessage(dstX, dstY, srcX, srcY, &k, sizeof(int64_t));
    // InterChiplet::sendMessage(dstX, dstY, srcX, srcY, &n, sizeof(int64_t));
    // InterChiplet::sendMessage(dstX, dstY, srcX, srcY, &a_size, sizeof(int64_t));
    // InterChiplet::sendMessage(dstX, dstY, srcX, srcY, &b_size, sizeof(int64_t));
    // InterChiplet::sendMessage(dstX, dstY, srcX, srcY, &c_size, sizeof(int64_t));
    std::cout << "sendMessage a_size: " << a_size*sizeof(float) << std::endl;
    std::cout << "sendMessage b_size: " << b_size*sizeof(float) << std::endl;
    std::cout << "sendMessage c_size: " << c_size*sizeof(float) << std::endl;
    std::cout << "batch_size: " << batch_size << std::endl;
    std::cout << "m: " << m << std::endl;
    std::cout << "k: " << k << std::endl;
    std::cout << "n: " << n << std::endl;
    std::cout << "batch_size*m*k: " << batch_size*m*k << std::endl;
    std::cout << "batch_size*k*n: " << batch_size*k*n << std::endl;
    // InterChiplet::sendMessage(dstX, dstY, srcX, srcY, &a_size, sizeof(int64_t));
    InterChiplet::sendMessage(dstX, dstY, srcX, srcY, a_array, a_size*sizeof(float));
    // InterChiplet::sendMessage(dstX, dstY, srcX, srcY, &b_size, sizeof(int64_t));
    InterChiplet::sendMessage(dstX, dstY, srcX, srcY, b_array, b_size*sizeof(float));
    // InterChiplet::sendMessage(dstX, dstY, srcX, srcY, &c_size, sizeof(int64_t));

    InterChiplet::receiveMessage(srcX, srcY, dstX, dstY, c_array, c_size*sizeof(float));
    // for(int64_t batch = 0; batch < batch_size; batch++){
    //     int64_t a_offset = batch * m * k;
    //     int64_t b_offset = batch * k * n;
    //     int64_t c_offset = batch * m * n;

    //     for(int64_t i = 0; i < m; i++){
    //         for(int64_t j = 0; j < n; j++){
    //             float sum = 0.0;
    //             for(int64_t p = 0; p < k; p++){
    //                 sum += a_array[a_offset + i * k + p] * b_array[b_offset + p * n + j];
    //             }
    //             c_array[c_offset + i * n + j] = sum;
    //         }
    //     }
    // }
    
    delete[] a_array;
    delete[] b_array;
    
    // 创建结果张量并复制数据
    c = torch::from_blob(c_array, c_shape, [c_array](void* p) { delete[] c_array; }, torch::kFloat32);
    
    // return c.clone();
}

/* 
    矩阵乘张量并行
    函数名：parallel_matmul()
    输入参数：
        torch::Tensor a: 第一个矩阵
        torch::Tensor b: 第二个矩阵
        int dim: 切分维度
        int src_x: 源矩阵x坐标
        int src_y: 源矩阵y坐标
        unordered_map<int, std::pair<int, int>> device_map: 设备映射
    输出参数：
        torch::Tensor c: 结果矩阵
*/

torch::Tensor parallel_matmul(torch::Tensor a, torch::Tensor b, int dim, int src_x, int src_y, std::unordered_map<int, std::pair<int, int>> device_map) {
    // 获取a和b的维度
    auto a_dim = a.sizes().vec();
    auto b_dim = b.sizes().vec();
    std::vector<int64_t> c_shape = a_dim;
    c_shape.back() = b_dim.back();  // 最后一维是乘法维度
    torch::Tensor c = torch::zeros(c_shape, a.options());  // 保持数据类型/设备一致

    int device_num = device_map.size();

    // 如果dim为-1，沿-1维度切分（纵向切分）
    if (dim == -1) {
        for(int i = 0; i < int(b_dim.size()); i++) {
            std::cout << "b_dim[" << i << "] = " << b_dim[i] << std::endl;
        }
        auto chunks_b = torch::chunk(b, device_num, dim);  // 沿第-1维切
        std::vector<torch::Tensor> b_list(chunks_b.begin(), chunks_b.end());

        std::vector<torch::Tensor> c_list(b_list.size());
        // 在创建线程前初始化张量
        for (size_t i = 0; i < b_list.size(); ++i) {
            // 创建正确形状的张量作为输出
            std::vector<int64_t> output_shape = a.sizes().vec();
            output_shape.back() = b_list[i].size(-1);
            c_list[i] = torch::zeros(output_shape, a.options());
        }

        // 然后创建线程
        std::vector<std::thread> threads;
        for (size_t i = 0; i < b_list.size(); ++i) {
            std::cout << "b_list[" << i << "] shape: " << b_list[i].sizes() << std::endl;
            threads.push_back(std::thread(custom_matmul_GPU, a, b_list[i], 
                     std::ref(c_list[i]), src_x, src_y, 
                     device_map[i].first, device_map[i].second));
        }
        for (auto& thread : threads) {
            thread.join();
        }
        std::cout<<"#########################"<<std::endl;

        c = torch::cat(c_list, dim);
    }

    return c;
}
// 等价于 Python 的 nn.ModuleList + deepcopy
template <typename Block> // 模板类，用于生成多个相同模块的列表
std::vector<std::shared_ptr<Block>> get_clones(std::shared_ptr<Block> module, int N) { // shared_ptr 智能指针
    std::vector<std::shared_ptr<Block>> clones;
    for (int i = 0; i < N; ++i) {
        clones.push_back(std::make_shared<Block>(*module));  // 深拷贝
    }
    return clones;
}

// Embedder 类
struct EmbedderImpl : torch::nn::Module {
    torch::nn::Embedding embed{nullptr};

    EmbedderImpl(int64_t vocab_size, int64_t d_model) {
        embed = register_module("embed", torch::nn::Embedding(vocab_size, d_model));
    }

    torch::Tensor forward(torch::Tensor x) {
        return embed->forward(x);
    }
};

TORCH_MODULE(Embedder); // 生成 Embedder 类型别名（简化用法：Embedder = std::shared_ptr<EmbedderImpl>）

// MultiHeadAttention 模块
struct MultiHeadAttentionImpl : torch::nn::Module {
    int64_t d_model;
    int64_t num_heads;
    int64_t head_dim;
    std::unordered_map<int, std::pair<int, int>> device_map;
    int srcX, srcY;

    torch::nn::Linear q_linear{nullptr}, k_linear{nullptr}, v_linear{nullptr}, out{nullptr};
    torch::nn::Dropout dropout{nullptr};

    MultiHeadAttentionImpl(int64_t num_heads, int64_t d_model, std::unordered_map<int, std::pair<int, int>> device_map, int srcX, int srcY, double dropout_rate = 0.1)
        : d_model(d_model), num_heads(num_heads), head_dim(d_model / num_heads), device_map(device_map), srcX(srcX), srcY(srcY) {
        
        q_linear = register_module("q_linear", torch::nn::Linear(d_model, d_model));
        k_linear = register_module("k_linear", torch::nn::Linear(d_model, d_model));
        v_linear = register_module("v_linear", torch::nn::Linear(d_model, d_model));
        out      = register_module("out", torch::nn::Linear(d_model, d_model));
        dropout  = register_module("dropout", torch::nn::Dropout(dropout_rate));
    }

    /*
        attention: [batch_size, num_heads, seq_len, head_dim] x [batch_size, num_heads, head_dim, seq_len] --> [batch_size, num_heads, seq_len, seq_len] --> softmax --> [batch_size, num_heads, seq_len, head_dim]
        q: [batch_size, num_heads, seq_len, head_dim]
        k: [batch_size, num_heads, seq_len, head_dim]
        v: [batch_size, num_heads, seq_len, head_dim]
        head_mask: [batch_size, num_heads, seq_len, seq_len]
        scores: [batch_size, num_heads, seq_len, seq_len]
        torch::matmul 要修改成手动切分张量，并分发给多个GPU（张量并行，分为横切和纵切）
        
        
        对于序列并行，每个GPU只会得到某一个序列的q，k，v。需要使用Ring Self-Attention 算法，将每个序列的k，v广播到每个GPU上。
        所以对于序列并行，每个GPU会得到对应序列的q，以及所有序列的k，v。
    */
    torch::Tensor attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor head_mask = {}) {
        // auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt((double)head_dim); // [batch_size, num_heads, seq_len, seq_len]
        auto scores = parallel_matmul(q, k.transpose(-2, -1), -1, srcX, srcY, device_map) / std::sqrt((double)head_dim);
        std::cout << "scores shape: " << scores.sizes() << std::endl;
        if (head_mask.defined()) {
            head_mask = head_mask.unsqueeze(1); // [batch_size, 1, seq_len, num_heads]
            scores = scores.masked_fill(head_mask == 0, -1e9);
        }

        scores = torch::softmax(scores, /*dim=*/-1);
        scores = dropout->forward(scores);
        // auto output = torch::matmul(scores, v); // [batch_size, num_heads, seq_len, head_dim]
        auto output = parallel_matmul(scores, v, -1, srcX, srcY, device_map);
        std::cout << "output shape: " << output.sizes() << std::endl;
        return output;
    }

    torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor head_mask = {}) {
        int64_t batch_size = q.size(0);
        int64_t seq_len = q.size(1);

        /* 
            张量并行（细粒度）：
                多头注意力并行 --> 张量并行（沿head_dim维度切分，每个注意力头下的矩阵乘计算分发给多个GPU）

        */
        q = q_linear->forward(q).view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2); // [batch_size, num_heads, seq_len, head_dim]
        k = k_linear->forward(k).view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
        v = v_linear->forward(v).view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);

        auto attn_out = attention(q, k, v, head_mask);  // [batch_size, num_heads, seq_len, head_dim]

        // Concatenate heads
        attn_out = attn_out.transpose(1, 2).contiguous().view({batch_size, seq_len, d_model}); // [batch_size, seq_len, d_model]

        return out->forward(attn_out);
    }
    /*
        序列并行：
                输入参数num_seq，num_seq代表seq_len维度切分数量，[batch_size,seq_len,head_dim] --> [batch_size,num_seq,seq_len,head_dim]
            沿num_seq维度切分：torch::chunk(tensor, dim) [batch_size,num_seq,seq_len,head_dim] --> [batch_size,1,seq_len,head_dim] --> [batch_size, seq_len, head_dim]

            序列并行（分Device） --> 多头注意力并行 --> 张量并行
    */
    torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, int64_t num_seq, torch::Tensor head_mask = {}){
        int64_t batch_size = q.size(0);
        int64_t seq_len = q.size(1);
        int64_t seq_dim = seq_len / num_seq;

        q = q_linear->forward(q).view({batch_size, num_seq, seq_dim, d_model});
        k = k_linear->forward(k).view({batch_size, num_seq, seq_dim, d_model});
        v = v_linear->forward(v).view({batch_size, num_seq, seq_dim, d_model});

        auto q_chunk = torch::chunk(q, 1);
        auto k_chunk = torch::chunk(k, 1);
        auto v_chunk = torch::chunk(v, 1);

        std::vector<torch::Tensor> q_vec(q_chunk.begin(), q_chunk.end());
        std::vector<torch::Tensor> k_vec(k_chunk.begin(), k_chunk.end());
        std::vector<torch::Tensor> v_vec(v_chunk.begin(), v_chunk.end());

        std::vector<torch::Tensor> attn_out;
        for(size_t i = 0; i < q_vec.size(); i++){
            attn_out.push_back(attention(q_vec[i], k_vec[i], v_vec[i], head_mask).unsqueeze(1));
        }
        torch::Tensor attention_out = torch::cat(attn_out, /*dim=*/1).view({batch_size, seq_len, d_model});
        return out->forward(attention_out);
    }
};

TORCH_MODULE(MultiHeadAttention);  // 生成别名：MultiHeadAttention = std::shared_ptr<MultiHeadAttentionImpl>

struct FeedForwardImpl : torch::nn::Module {
    torch::nn::Linear linear1{nullptr}, linear2{nullptr};
    torch::nn::Dropout dropout{nullptr};

    FeedForwardImpl(int64_t d_model, int64_t hidden_dim, double dropout_rate = 0.1) {
        linear1 = register_module("linear1", torch::nn::Linear(d_model, hidden_dim));
        linear2 = register_module("linear2", torch::nn::Linear(hidden_dim, d_model));
        dropout = register_module("dropout", torch::nn::Dropout(dropout_rate));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = dropout->forward(torch::relu(linear1->forward(x)));
        x = linear2->forward(x);
        return x;
    }
};

TORCH_MODULE(FeedForward);  // 别名：FeedForward = std::shared_ptr<FeedForwardImpl>

struct NormImpl : torch::nn::Module {
    torch::nn::LayerNorm layer_norm{nullptr};

    NormImpl(int64_t d_model, double eps = 1e-6) {
        layer_norm = register_module("layer_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model}).eps(eps)));
    }

    torch::Tensor forward(torch::Tensor x) {
        return layer_norm->forward(x);
    }
};
TORCH_MODULE(Norm);  // 别名：Norm = std::shared_ptr<NormImpl>

// EncoderLayer 实现
// 对于序列并行，在EncoderLayer中就要拆分x到多个GPU上
struct EncoderLayerImpl : torch::nn::Module {
    Norm norm_1, norm_2;
    MultiHeadAttention attn;
    FeedForward ff;
    torch::nn::Dropout dropout_1{nullptr}, dropout_2{nullptr};

    EncoderLayerImpl(int64_t d_model, int64_t heads, std::unordered_map<int, std::pair<int, int>> device_map, int srcX, int srcY, double dropout = 0.1)
        : norm_1(d_model), norm_2(d_model),
          attn(MultiHeadAttention(heads, d_model, device_map, srcX, srcY, dropout)),
          ff(FeedForward(d_model, d_model * 4, dropout)) {
        dropout_1 = register_module("dropout_1", torch::nn::Dropout(dropout));
        dropout_2 = register_module("dropout_2", torch::nn::Dropout(dropout));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor mask) {
        auto attn_output = dropout_1->forward(attn->forward(x, x, x, mask));
        x = norm_1->forward(x + attn_output);
        auto ff_output = dropout_2->forward(ff->forward(x));
        x = norm_2->forward(x + ff_output);
        return x;
    }
};
TORCH_MODULE(EncoderLayer);

// PositionalEncoder 实现
struct PositionalEncoderImpl : torch::nn::Module {
    torch::Tensor pe;
    int64_t d_model;

    PositionalEncoderImpl(int64_t d_model, int64_t max_seq_len = max_seq_len) : d_model(d_model) {
        pe = torch::zeros({max_seq_len, d_model});
        for (int64_t pos = 0; pos < max_seq_len; ++pos) {
            for (int64_t i = 0; i < d_model; i += 2) {
                pe[pos][i] = std::sin(pos / std::pow(10000.0, i / (double)d_model));
                if (i + 1 < d_model)
                    pe[pos][i + 1] = std::cos(pos / std::pow(10000.0, i / (double)d_model));
            }
        }
        pe = pe.unsqueeze(0); // [1, max_seq_len, d_model]
        register_buffer("pe", pe);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = x * std::sqrt((double)d_model);
        auto seq_len = x.size(1);
        return x + pe.slice(1, 0, seq_len);
    }
};
TORCH_MODULE(PositionalEncoder);

// Encoder 实现
struct EncoderImpl : torch::nn::Module {
    int64_t N;
    Embedder embed;
    PositionalEncoder pe;
    std::vector<EncoderLayer> layers;
    Norm norm;

    EncoderImpl(int64_t vocab_size, int64_t d_model, int64_t N, int64_t heads, std::unordered_map<int, std::pair<int, int>> device_map, int srcX, int srcY, double dropout = 0.1)
        : N(N), embed(Embedder(vocab_size, d_model)), pe(PositionalEncoder(d_model)), norm(Norm(d_model)) {
        for (int64_t i = 0; i < N; ++i)
            layers.push_back(EncoderLayer(d_model, heads, device_map, srcX, srcY, dropout));
    }

    torch::Tensor forward(torch::Tensor src, torch::Tensor mask) {
        auto x = embed->forward(src);
        x = pe->forward(x);
        for (auto& layer : layers)
            x = layer->forward(x, mask);
        return norm->forward(x);
    }
};
TORCH_MODULE(Encoder);

// DecoderLayer 实现
struct DecoderLayerImpl : torch::nn::Module {
    Norm norm_1, norm_2, norm_3;
    MultiHeadAttention attn_1, attn_2;
    FeedForward ff;
    torch::nn::Dropout dropout_1{nullptr}, dropout_2{nullptr}, dropout_3{nullptr};

    DecoderLayerImpl(int64_t d_model, int64_t heads, std::unordered_map<int, std::pair<int, int>> device_map, int srcX, int srcY, double dropout = 0.1)
        : norm_1(d_model), norm_2(d_model), norm_3(d_model),
          attn_1(MultiHeadAttention(heads, d_model, device_map, srcX, srcY, dropout)),
          attn_2(MultiHeadAttention(heads, d_model, device_map, srcX, srcY, dropout)),
          ff(FeedForward(d_model, d_model * 4, dropout)) {
        dropout_1 = register_module("dropout_1", torch::nn::Dropout(dropout));
        dropout_2 = register_module("dropout_2", torch::nn::Dropout(dropout));
        dropout_3 = register_module("dropout_3", torch::nn::Dropout(dropout));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor e_outputs, torch::Tensor src_mask, torch::Tensor trg_mask) {
        x = norm_1->forward(x + dropout_1->forward(attn_1->forward(x, x, x, trg_mask)));
        x = norm_2->forward(x + dropout_2->forward(attn_2->forward(x, e_outputs, e_outputs, src_mask)));
        x = norm_3->forward(x + dropout_3->forward(ff->forward(x)));
        return x;
    }
};
TORCH_MODULE(DecoderLayer);

// Decoder 实现
struct DecoderImpl : torch::nn::Module {
    int64_t N;
    Embedder embed;
    PositionalEncoder pe;
    std::vector<DecoderLayer> layers;
    Norm norm;

    DecoderImpl(int64_t vocab_size, int64_t d_model, int64_t N, int64_t heads, std::unordered_map<int, std::pair<int, int>> device_map, int srcX, int srcY, double dropout = 0.1)
        : N(N), embed(Embedder(vocab_size, d_model)), pe(PositionalEncoder(d_model)), norm(Norm(d_model)) {
        for (int64_t i = 0; i < N; ++i)
            layers.push_back(DecoderLayer(d_model, heads, device_map, srcX, srcY, dropout));
    }

    torch::Tensor forward(torch::Tensor trg, torch::Tensor e_outputs, torch::Tensor src_mask, torch::Tensor trg_mask) {
        auto x = embed->forward(trg);
        x = pe->forward(x);
        for (int64_t i = 0; i < N; ++i)
            x = layers[i]->forward(x, e_outputs, src_mask, trg_mask);
        return norm->forward(x);
    }
};
TORCH_MODULE(Decoder);

// Transformer 顶层封装
struct TransformerImpl : torch::nn::Module {
    Encoder encoder;
    Decoder decoder;
    torch::nn::Linear out{nullptr};

    TransformerImpl(int64_t src_vocab, int64_t trg_vocab, int64_t d_model, int64_t N, int64_t heads, std::unordered_map<int, std::pair<int, int>> device_map, int srcX, int srcY, double dropout = 0.1)
        : encoder(Encoder(src_vocab, d_model, N, heads, device_map, srcX, srcY, dropout)),
          decoder(Decoder(trg_vocab, d_model, N, heads, device_map, srcX, srcY, dropout)) {
        
        out = register_module("out", torch::nn::Linear(d_model, trg_vocab));
        register_module("encoder", encoder);
        register_module("decoder", decoder);
    }

    torch::Tensor forward(torch::Tensor src, torch::Tensor trg, torch::Tensor src_mask, torch::Tensor trg_mask) {
        auto e_outputs = encoder->forward(src, src_mask);
        auto d_output = decoder->forward(trg, e_outputs, src_mask, trg_mask);
        return out->forward(d_output);
    }
};

TORCH_MODULE(Transformer);

bool readCSV(std::string csv_path, std::unordered_map<int, std::pair<int, int>> &device_map, int &device_num, int &cpu_node, int topology_width) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << csv_path << std::endl;
        return false;
    }

    std::string line;
    int device_index = 0;
    device_num = 0;  // 重置device_num

    // 读取标题行
    std::getline(file, line);

    // 读取数据行
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string node_str, device, clock;
        
        std::getline(ss, node_str, ',');
        std::getline(ss, device, ',');
        std::getline(ss, clock, ',');
        
        // 移除可能存在的空白字符
        node_str.erase(std::remove_if(node_str.begin(), node_str.end(), ::isspace), node_str.end());
        device.erase(std::remove_if(device.begin(), device.end(), ::isspace), device.end());
        
        try {
            int node = std::stoi(node_str);
            
            if (device == "CPU") {
                cpu_node = node;
            } else {
                std::cout << "node: " << node << std::endl;
                int x = node / topology_width;
                int y = node % topology_width;
                device_map[device_index++] = std::make_pair(x, y);
                device_num++;
            }
        } catch (const std::exception& e) {
            std::cerr << "解析CSV行时出错: " << line << std::endl;
            std::cerr << "错误: " << e.what() << std::endl;
            continue;  // 跳过这一行
        }
    }

    file.close();

    // 打印调试信息
    std::cout << "读取CSV文件: " << csv_path << std::endl;
    std::cout << "CPU节点: " << cpu_node << std::endl;
    std::cout << "设备数量: " << device_num << std::endl;

    // 打印GPU映射
    std::cout << "GPU映射:" << std::endl;
    for (const auto& pair : device_map) {
        std::cout << pair.first << " -> (" << pair.second.first << ", " << pair.second.second << ")" << std::endl;
    }

    return true;
}

int main(int argc, const char* argv[]) {
    CmdLineOptions opt;
    if (opt.parse(argc, argv) != 0) {
        return 0;
    };
    const int64_t d_model = opt.d_model;
    const int64_t heads = opt.num_heads;
    const int64_t N = opt.N;
    const int64_t src_vocab = opt.src_vocab;
    const int64_t trg_vocab = opt.trg_vocab;
    const int64_t batch_size = opt.batch_size;
    const int64_t seq_len = opt.seq_len;
    const double dropout = opt.dropout;
    const int topology_width = opt.m_topology_width;
    // const int topology_height = 2;

    max_seq_len = seq_len;

    std::unordered_map<int, std::pair<int, int>> device_map;
    int device_num = 0;
    int cpu_node = -1;
    
    try {
        if (!readCSV("../../mapDevice.csv", device_map, device_num, cpu_node, topology_width)) {
            std::cerr << "读取CSV文件失败" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "读取CSV文件失败: " << e.what() << std::endl;
        return 1;
    }

    int srcX = opt.m_srcX;
    int srcY = opt.m_srcY;

    // 使用randn创建Float类型张量，或者显式转换类型
    torch::Tensor a = torch::randn({10, 10, 10, 10}, torch::kFloat32);
    torch::Tensor b = torch::randn({10, 10, 10, 5}, torch::kFloat32);
    
    // 或者保持使用randint但显式转换类型
    // torch::Tensor a = torch::randint(1, 100, {10, 10, 10, 10}).to(torch::kFloat32);
    // torch::Tensor b = torch::randint(1, 100, {10, 10, 10, 5}).to(torch::kFloat32);
    
    // auto c = parallel_matmul(a, b, -1, srcX, srcY, device_map);
    // std::cout << "c shape: " << c.sizes() << std::endl;
    
    // 创建模型
    Transformer model(src_vocab, trg_vocab, d_model, N, heads, device_map, srcX, srcY, dropout);
    model->eval(); // 推理模式

    // 随机输入
    auto src = torch::randint(0, src_vocab, {batch_size, seq_len}, torch::kLong);
    auto trg = torch::randint(0, trg_vocab, {batch_size, seq_len}, torch::kLong);

    auto src_mask = torch::ones({batch_size, 1, seq_len});
    auto trg_mask = torch::tril(torch::ones({batch_size, seq_len, seq_len}));

    auto output = model->forward(src, trg, src_mask, trg_mask);
    
    bool is_end = true;
    for(int i = 0; i < device_num; i++){
        int dstX = device_map[i].first;
        int dstY = device_map[i].second;
        InterChiplet::sendMessage(dstX, dstY, srcX, srcY,  &is_end, sizeof(bool));
    }

    std::cout << "Output shape: " << output.sizes() << std::endl;

    return 0;
}
