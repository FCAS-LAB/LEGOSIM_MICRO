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
#include <random>
#include <cassert>
#include "apis_c.h"
#include "sim_api.h"

#define embedding_comm_id 1
#define mlp_comm_id 0

// 简单的张量类替代torch::Tensor
class Tensor {
public:
    std::vector<float> data;
    std::vector<int64_t> shape;
    int64_t numel;
    
    Tensor() : numel(0) {}
    
    Tensor(const std::vector<int64_t>& shape) : shape(shape) {
        numel = 1;
        for (auto s : shape) {
            numel *= s;
        }
        data.resize(numel, 0.0f);
    }
    
    Tensor(const std::vector<int64_t>& shape, float value) : shape(shape) {
        numel = 1;
        for (auto s : shape) {
            numel *= s;
        }
        data.resize(numel, value);
    }
    
    // 从原始数据构造
    Tensor(const std::vector<int64_t>& shape, const std::vector<float>& data) 
        : shape(shape), data(data) {
        numel = 1;
        for (auto s : shape) {
            numel *= s;
        }
        assert(numel == data.size());
    }
    
    // 获取形状
    std::vector<int64_t> sizes() const { return shape; }
    int64_t size(int dim) const { return shape[dim]; }
    
    // 索引访问
    float& operator[](int64_t idx) { return data[idx]; }
    const float& operator[](int64_t idx) const { return data[idx]; }
    
    // 多维索引
    float& at(const std::vector<int64_t>& indices) {
        int64_t idx = 0;
        int64_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            idx += indices[i] * stride;
            stride *= shape[i];
        }
        return data[idx];
    }
    
    const float& at(const std::vector<int64_t>& indices) const {
        int64_t idx = 0;
        int64_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            idx += indices[i] * stride;
            stride *= shape[i];
        }
        return data[idx];
    }
    
    // 修复：重新调整形状
    Tensor view(const std::vector<int64_t>& new_shape) const {
        Tensor result;
        result.data = this->data;
        
        // 处理-1维度的自动推断
        std::vector<int64_t> fixed_shape = new_shape;
        int64_t total_known = 1;
        int auto_dim_index = -1;
        
        for (size_t i = 0; i < new_shape.size(); i++) {
            if (new_shape[i] == -1) {
                if (auto_dim_index != -1) {
                    throw std::runtime_error("只能有一个维度为-1");
                }
                auto_dim_index = i;
            } else {
                total_known *= new_shape[i];
            }
        }
        
        if (auto_dim_index != -1) {
            fixed_shape[auto_dim_index] = this->numel / total_known;
        }
        
        result.shape = fixed_shape;
        result.numel = this->numel;
        return result;
    }
    
    // 转置
    Tensor transpose(int dim1, int dim2) const {
        std::vector<int64_t> new_shape = shape;
        std::swap(new_shape[dim1], new_shape[dim2]);
        
        Tensor result(new_shape);
        
        if (shape.size() == 2 && dim1 == 0 && dim2 == 1) {
            // 2D转置
            for (int64_t i = 0; i < shape[0]; i++) {
                for (int64_t j = 0; j < shape[1]; j++) {
                    result.at({j, i}) = this->at({i, j});
                }
            }
        } else if (shape.size() == 3 && dim1 == 1 && dim2 == 2) {
            // 3D转置 (batch, dim1, dim2) -> (batch, dim2, dim1)
            for (int64_t b = 0; b < shape[0]; b++) {
                for (int64_t i = 0; i < shape[1]; i++) {
                    for (int64_t j = 0; j < shape[2]; j++) {
                        result.at({b, j, i}) = this->at({b, i, j});
                    }
                }
            }
        }
        return result;
    }
    
    // 修复：连接操作
    static Tensor cat(const std::vector<Tensor>& tensors, int dim) {
        if (tensors.empty()) return Tensor();
        
        std::vector<int64_t> new_shape = tensors[0].shape;
        for (size_t i = 1; i < tensors.size(); i++) {
            new_shape[dim] += tensors[i].shape[dim];
        }
        
        Tensor result(new_shape);
        
        if (dim == 1 && tensors[0].shape.size() == 2) {
            // 2D张量在dim=1上连接
            int64_t offset = 0;
            for (const auto& tensor : tensors) {
                for (int64_t i = 0; i < tensor.shape[0]; i++) {
                    for (int64_t j = 0; j < tensor.shape[1]; j++) {
                        result.at({i, offset + j}) = tensor.at({i, j});
                    }
                }
                offset += tensor.shape[1];
            }
        } else if (dim == tensors[0].shape.size() - 1) {
            // 在最后一个维度连接
            int64_t offset = 0;
            for (const auto& tensor : tensors) {
                std::copy(tensor.data.begin(), tensor.data.end(), 
                         result.data.begin() + offset);
                offset += tensor.numel;
            }
        } else if (dim == 0) {
            // 在第一个维度连接
            int64_t offset = 0;
            int64_t stride = tensors[0].numel / tensors[0].shape[0];
            for (const auto& tensor : tensors) {
                std::copy(tensor.data.begin(), tensor.data.end(),
                         result.data.begin() + offset);
                offset += tensor.numel;
            }
        }
        return result;
    }
    
    // 分块操作
    std::vector<Tensor> chunk(int chunks, int dim) const {
        std::vector<Tensor> result;
        int64_t chunk_size = shape[dim] / chunks;
        int64_t remainder = shape[dim] % chunks;
        
        int64_t current_start = 0;
        for (int i = 0; i < chunks; i++) {
            int64_t current_chunk_size = chunk_size + (i < remainder ? 1 : 0);
            
            std::vector<int64_t> new_shape = shape;
            new_shape[dim] = current_chunk_size;
            Tensor chunk_tensor(new_shape);
            
            if (dim == 0) {
                // 在第一个维度分块
                int64_t stride = numel / shape[0];
                int64_t start = current_start * stride;
                int64_t size = current_chunk_size * stride;
                std::copy(data.begin() + start, data.begin() + start + size,
                         chunk_tensor.data.begin());
            }
            
            result.push_back(chunk_tensor);
            current_start += current_chunk_size;
        }
        return result;
    }
    
    // 矩阵乘法
    Tensor mm(const Tensor& other) const {
        assert(shape.size() == 2 && other.shape.size() == 2);
        assert(shape[1] == other.shape[0]);
        
        Tensor result({shape[0], other.shape[1]});
        
        for (int64_t i = 0; i < shape[0]; i++) {
            for (int64_t j = 0; j < other.shape[1]; j++) {
                float sum = 0.0f;
                for (int64_t k = 0; k < shape[1]; k++) {
                    sum += this->at({i, k}) * other.at({k, j});
                }
                result.at({i, j}) = sum;
            }
        }
        return result;
    }
    
    // 批次矩阵乘法
    Tensor bmm(const Tensor& other) const {
        assert(shape.size() == 3 && other.shape.size() == 3);
        assert(shape[0] == other.shape[0]);
        assert(shape[2] == other.shape[1]);
        
        Tensor result({shape[0], shape[1], other.shape[2]});
        
        for (int64_t b = 0; b < shape[0]; b++) {
            for (int64_t i = 0; i < shape[1]; i++) {
                for (int64_t j = 0; j < other.shape[2]; j++) {
                    float sum = 0.0f;
                    for (int64_t k = 0; k < shape[2]; k++) {
                        sum += this->at({b, i, k}) * other.at({b, k, j});
                    }
                    result.at({b, i, j}) = sum;
                }
            }
        }
        return result;
    }
    
    // 获取数据指针
    float* data_ptr() { return data.data(); }
    const float* data_ptr() const { return data.data(); }
};

// 整数张量类
class IntTensor {
public:
    std::vector<int64_t> data;
    std::vector<int64_t> shape;
    int64_t numel;
    
    IntTensor() : numel(0) {}
    
    IntTensor(const std::vector<int64_t>& shape) : shape(shape) {
        numel = 1;
        for (auto s : shape) {
            numel *= s;
        }
        data.resize(numel, 0);
    }
    
    IntTensor(const std::vector<int64_t>& shape, const std::vector<int64_t>& data) 
        : shape(shape), data(data) {
        numel = 1;
        for (auto s : shape) {
            numel *= s;
        }
    }
    
    std::vector<int64_t> sizes() const { return shape; }
    int64_t size(int dim) const { return shape[dim]; }
    
    int64_t& operator[](int64_t idx) { return data[idx]; }
    const int64_t& operator[](int64_t idx) const { return data[idx]; }
    
    int64_t* data_ptr() { return data.data(); }
    const int64_t* data_ptr() const { return data.data(); }
};

// 随机数生成器
class RandomGenerator {
private:
    static std::random_device rd;
    static std::mt19937 gen;
    static std::normal_distribution<float> normal_dist;
    static std::uniform_real_distribution<float> uniform_dist;
    static std::uniform_int_distribution<int64_t> int_dist;
    
public:
    static Tensor randn(const std::vector<int64_t>& shape) {
        Tensor result(shape);
        for (auto& val : result.data) {
            val = normal_dist(gen);
        }
        return result;
    }
    
    static Tensor rand(const std::vector<int64_t>& shape) {
        Tensor result(shape);
        for (auto& val : result.data) {
            val = uniform_dist(gen);
        }
        return result;
    }
    
    static Tensor uniform(const std::vector<int64_t>& shape, float low, float high) {
        Tensor result(shape);
        std::uniform_real_distribution<float> dist(low, high);
        for (auto& val : result.data) {
            val = dist(gen);
        }
        return result;
    }
    
    static IntTensor randint(int64_t low, int64_t high, const std::vector<int64_t>& shape) {
        IntTensor result(shape);
        std::uniform_int_distribution<int64_t> dist(low, high - 1);
        for (auto& val : result.data) {
            val = dist(gen);
        }
        return result;
    }
};

// 静态成员定义
std::random_device RandomGenerator::rd;
std::mt19937 RandomGenerator::gen(RandomGenerator::rd());
std::normal_distribution<float> RandomGenerator::normal_dist(0.0, 1.0);
std::uniform_real_distribution<float> RandomGenerator::uniform_dist(0.0, 1.0);
std::uniform_int_distribution<int64_t> RandomGenerator::int_dist;

// 工具函数
Tensor ones(const std::vector<int64_t>& shape) {
    return Tensor(shape, 1.0f);
}

Tensor zeros(const std::vector<int64_t>& shape) {
    return Tensor(shape, 0.0f);
}

Tensor empty(const std::vector<int64_t>& shape) {
    return Tensor(shape);
}

IntTensor tensor_int(const std::vector<int64_t>& data) {
    return IntTensor({static_cast<int64_t>(data.size())}, data);
}

// 激活函数
namespace activation {
    void relu_inplace(Tensor& x) {
        for (auto& val : x.data) {
            val = std::max(0.0f, val);
        }
    }
    
    void sigmoid_inplace(Tensor& x) {
        for (auto& val : x.data) {
            val = 1.0f / (1.0f + std::exp(-val));
        }
    }
    
    Tensor relu(const Tensor& x) {
        Tensor result = x;
        relu_inplace(result);
        return result;
    }
    
    Tensor sigmoid(const Tensor& x) {
        Tensor result = x;
        sigmoid_inplace(result);
        return result;
    }
}

// 线性层
class Linear {
public:
    Tensor weight;
    Tensor bias;
    bool has_bias;
    
    Linear(int64_t in_features, int64_t out_features, bool bias = true) 
        : has_bias(bias) {
        
        // 初始化权重
        float std_dev = std::sqrt(2.0f / (in_features + out_features));
        weight = RandomGenerator::randn({out_features, in_features});
        for (auto& val : weight.data) {
            val = val * std_dev;
        }
        
        if (has_bias) {
            // 初始化偏置
            std_dev = std::sqrt(1.0f / out_features);
            this->bias = RandomGenerator::randn({out_features});
            for (auto& val : this->bias.data) {
                val = val * std_dev;
            }
        }
    }
    
    Tensor forward(const Tensor& input) const {
        // input: [batch_size, in_features]
        // weight: [out_features, in_features]
        // output: [batch_size, out_features]
        
        Tensor result = input.mm(weight.transpose(0, 1));
        
        if (has_bias) {
            // 广播加法
            for (int64_t i = 0; i < result.shape[0]; i++) {
                for (int64_t j = 0; j < result.shape[1]; j++) {
                    result.at({i, j}) += bias[j];
                }
            }
        }
        return result;
    }
};

// 嵌入包层
class EmbeddingBag {
public:
    Tensor weight;
    int64_t num_embeddings;
    int64_t embedding_dim;
    
    EmbeddingBag(int64_t num_embeddings, int64_t embedding_dim) 
        : num_embeddings(num_embeddings), embedding_dim(embedding_dim) {
        
        float bound = std::sqrt(1.0f / num_embeddings);
        weight = RandomGenerator::uniform({num_embeddings, embedding_dim}, -bound, bound);
    }
    
    Tensor forward(const IntTensor& indices, const IntTensor& offsets, 
                   const Tensor* per_sample_weights = nullptr) const {
        
        int64_t batch_size = offsets.shape[0];
        Tensor result({batch_size, embedding_dim}, 0.0f);
        
        for (int64_t b = 0; b < batch_size; b++) {
            int64_t start = (b > 0) ? offsets[b-1] : 0;
            int64_t end = offsets[b];
            
            for (int64_t i = start; i < end; i++) {
                int64_t idx = indices[i];
                float weight_val = per_sample_weights ? (*per_sample_weights)[i] : 1.0f;
                
                for (int64_t d = 0; d < embedding_dim; d++) {
                    result.at({b, d}) += weight.at({idx, d}) * weight_val;
                }
            }
        }
        
        return result;
    }
};

// MLP序列
class MLP {
public:
    std::vector<std::unique_ptr<Linear>> layers;
    std::vector<int> activation_types; // 0: none, 1: relu, 2: sigmoid
    
    void add_layer(std::unique_ptr<Linear> layer, int activation_type = 1) {
        layers.push_back(std::move(layer));
        activation_types.push_back(activation_type);
    }
    
    Tensor forward(const Tensor& input) const {
        Tensor x = input;
        
        for (size_t i = 0; i < layers.size(); i++) {
            x = layers[i]->forward(x);
            
            if (activation_types[i] == 1) {
                activation::relu_inplace(x);
            } else if (activation_types[i] == 2) {
                activation::sigmoid_inplace(x);
            }
        }
        
        return x;
    }
    
    size_t size() const { return layers.size(); }
};

// 主要的DLRM实现类
class DLRMImpl {
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
    
    std::vector<std::unique_ptr<EmbeddingBag>> emb_l;
    std::unique_ptr<MLP> bot_l, top_l;
    bool quantize_emb;
    int64_t quantize_bits;
    std::vector<Tensor> v_W_l;
    std::vector<std::unique_ptr<MLP>> bot_l_replicas, top_l_replicas;
    int src_x, src_y;
    std::unordered_map<int, std::pair<int, int>> device_map;

    /*
        ln: 每层神经元数量列表
        sigmoid_layer: 应用sigmoid激活函数的层索引
    */
    std::unique_ptr<MLP> create_mlp(const std::vector<int64_t>& ln, int64_t sigmoid_layer) {
        auto mlp = std::make_unique<MLP>();
        
        for (size_t i = 0; i < ln.size() - 1; i++) {
            int64_t n = ln[i];
            int64_t m = ln[i + 1];
            
            auto linear = std::make_unique<Linear>(n, m, true);
            
            int activation_type = (i == sigmoid_layer) ? 2 : 1; // 2: sigmoid, 1: relu
            mlp->add_layer(std::move(linear), activation_type);
        }
        
        return mlp;
    }

    std::pair<std::vector<std::unique_ptr<EmbeddingBag>>, std::vector<Tensor>> 
    create_emb(int64_t m, const std::vector<int64_t>& ln, const std::string& weighted_pooling = "") {
        
        std::vector<std::unique_ptr<EmbeddingBag>> emb_l;
        std::vector<Tensor> v_W_l;
        
        for (size_t i = 0; i < ln.size(); i++) {
            int64_t n = ln[i];
            auto emb = std::make_unique<EmbeddingBag>(n, m);
            emb_l.push_back(std::move(emb));
            
            if (weighted_pooling == "") {
                v_W_l.push_back(Tensor()); // 空张量
            } else {
                v_W_l.push_back(ones({n}));
            }
        }
        
        return {std::move(emb_l), std::move(v_W_l)};
    }

    Tensor interact_features(const Tensor& x, const std::vector<Tensor>& ly) {
        if (arch_interaction_op == "dot") {
            // 获取batch大小和维度
            int64_t batch_size = x.size(0);
            int64_t d = x.size(1);
            
            std::cout << "interact_features: batch_size=" << batch_size << ", d=" << d << std::endl;
            std::cout << "x shape: [" << x.size(0) << ", " << x.size(1) << "]" << std::endl;
            
            // 连接密集特征和稀疏特征
            std::vector<Tensor> concat_list = {x};
            concat_list.insert(concat_list.end(), ly.begin(), ly.end());
            
            std::cout << "concat_list size: " << concat_list.size() << std::endl;
            for (size_t i = 0; i < concat_list.size(); i++) {
                std::cout << "concat_list[" << i << "] shape: [" << concat_list[i].size(0) << ", " << concat_list[i].size(1) << "]" << std::endl;
            }
            
            Tensor T_cat = Tensor::cat(concat_list, 1);
            std::cout << "T_cat shape: [" << T_cat.size(0) << ", " << T_cat.size(1) << "]" << std::endl;
            
            // 计算特征数量
            int64_t num_features = 1 + ly.size();  // 1个密集特征 + ly.size()个稀疏特征
            Tensor T = T_cat.view({batch_size, num_features, d});
            std::cout << "T shape after view: [" << T.size(0) << ", " << T.size(1) << ", " << T.size(2) << "]" << std::endl;
            
            // 执行点积操作
            Tensor T_transpose = T.transpose(1, 2);
            std::cout << "T_transpose shape: [" << T_transpose.size(0) << ", " << T_transpose.size(1) << ", " << T_transpose.size(2) << "]" << std::endl;
            
            Tensor Z = T.bmm(T_transpose);
            std::cout << "Z shape: [" << Z.size(0) << ", " << Z.size(1) << ", " << Z.size(2) << "]" << std::endl;
            
            // 提取下三角部分
            auto sizes = Z.sizes();
            int64_t ni = sizes[1];
            int64_t nj = sizes[2];
            
            // 创建索引
            int64_t offset = arch_interaction_itself ? 1 : 0;
            std::vector<int64_t> li_vec, lj_vec;
            
            for (int64_t i = 0; i < ni; i++) {
                for (int64_t j = 0; j < i + offset; j++) {
                    li_vec.push_back(i);
                    lj_vec.push_back(j);
                }
            }
            
            std::cout << "Lower triangle indices count: " << li_vec.size() << std::endl;
            
            // 提取特定的索引位置
            Tensor Zflat({batch_size, static_cast<int64_t>(li_vec.size())});
            for (int64_t b = 0; b < batch_size; b++) {
                for (size_t k = 0; k < li_vec.size(); k++) {
                    Zflat.at({b, static_cast<int64_t>(k)}) = Z.at({b, li_vec[k], lj_vec[k]});
                }
            }
            
            std::cout << "Zflat shape: [" << Zflat.size(0) << ", " << Zflat.size(1) << "]" << std::endl;
            
            // 连接密集特征和交互特征
            return Tensor::cat({x, Zflat}, 1);
        } 
        else if (arch_interaction_op == "cat") {
            // 简单连接所有特征
            std::vector<Tensor> concat_list = {x};
            concat_list.insert(concat_list.end(), ly.begin(), ly.end());
            return Tensor::cat(concat_list, 1);
        } 
        else {
            throw std::runtime_error("ERROR: --arch-interaction-op=" + arch_interaction_op + " is not supported");
        }
    }

    // 复制模块到多个设备（简化版）
    std::vector<std::unique_ptr<MLP>> replicate_modules(
        const MLP& mlp, 
        const std::vector<int64_t>& device_ids) {
        
        std::vector<std::unique_ptr<MLP>> replicas;
        for (size_t i = 0; i < device_ids.size(); i++) {
            auto replica = std::make_unique<MLP>();
            
            // 复制层结构和参数
            for (size_t j = 0; j < mlp.layers.size(); j++) {
                auto& original_layer = mlp.layers[j];
                auto new_layer = std::make_unique<Linear>(
                    original_layer->weight.size(1), 
                    original_layer->weight.size(0), 
                    original_layer->has_bias
                );
                
                // 复制权重和偏置
                new_layer->weight = original_layer->weight;
                if (original_layer->has_bias) {
                    new_layer->bias = original_layer->bias;
                }
                
                replica->add_layer(std::move(new_layer), mlp.activation_types[j]);
            }
            
            replicas.push_back(std::move(replica));
        }
        return replicas;
    }

    // 在多个模块上并行应用输入
    std::vector<Tensor> parallel_apply_modules(
        std::vector<std::unique_ptr<MLP>>& mlps,
        std::vector<Tensor>& inputs) {
        
        std::vector<Tensor> outputs(mlps.size());
        std::vector<std::thread> threads;
        
        for (size_t i = 0; i < mlps.size(); i++) {
            threads.emplace_back([&, i]() {
                outputs[i] = mlps[i]->forward(inputs[i]);
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        return outputs;
    }

    // 将张量分散到多个设备
    std::vector<Tensor> scatter_tensors(
        const Tensor& tensor, 
        const std::vector<int64_t>& device_ids,
        int64_t dim = 0) {
        
        return tensor.chunk(device_ids.size(), dim);
    }

    // 收集多个设备上的张量
    Tensor gather_tensors(
        const std::vector<Tensor>& tensors, 
        int64_t dim = 0) {
        
        return Tensor::cat(tensors, dim);
    }

    void embeddingbag_forward_cuda(
        float* weight,
        int64_t* indices,
        int64_t* offsets,
        float* per_sample_weights,
        float* output,
        int64_t batch_size,
        int64_t num_embeddings,
        int64_t embedding_dim,
        int64_t num_indices,
        int64_t src_x,
        int64_t src_y,
        int64_t dst_x,
        int64_t dst_y
    ) {
        // ... 保持原有的CUDA通信代码 ...
        int64_t weight_size = num_embeddings * embedding_dim * sizeof(float);
        int64_t indices_size = num_indices * sizeof(int64_t);
        int64_t offsets_size = batch_size * sizeof(int64_t);
        int64_t output_size = batch_size * embedding_dim * sizeof(float);
        
        std::cout << "batch_size: " << batch_size << std::endl;
        std::cout << "num_embeddings: " << num_embeddings << std::endl;
        std::cout << "embedding_dim: " << embedding_dim << std::endl;
        std::cout << "num_indices: " << num_indices << std::endl;
        std::cout << "weight_size: " << weight_size << std::endl;
        std::cout << "indices_size: " << indices_size << std::endl;
        std::cout << "offsets_size: " << offsets_size << std::endl;
        std::cout << "output_size: " << output_size << std::endl;

        bool is_run = true;
        int64_t module_id = embedding_comm_id;
        InterChiplet::sendMessage(dst_x, dst_y, src_x, src_y, &is_run, sizeof(bool));
        InterChiplet::sendMessage(dst_x, dst_y, src_x, src_y, &module_id, sizeof(int64_t));
        std::cout << "--------------embeddingbag_forward_cuda start--------------------" << std::endl;

        int64_t sizeList[4] = {num_embeddings, num_indices, batch_size, embedding_dim};
        InterChiplet::sendMessage(dst_x, dst_y, src_x, src_y, sizeList, 4 * sizeof(int64_t));
        
        InterChiplet::sendMessage(dst_x, dst_y, src_x, src_y, weight, weight_size);
        InterChiplet::sendMessage(dst_x, dst_y, src_x, src_y, indices, indices_size);
        InterChiplet::sendMessage(dst_x, dst_y, src_x, src_y, offsets, offsets_size);
        
        InterChiplet::receiveMessage(src_x, src_y, dst_x, dst_y, output, output_size);
        std::cout << "--------------embeddingbag_forward_cuda end--------------------" << std::endl;
    }

    // 应用嵌入操作
    std::vector<Tensor> apply_emb(
        const IntTensor& lS_o,
        const std::vector<IntTensor>& lS_i,
        std::vector<std::unique_ptr<EmbeddingBag>>& emb_l,
        std::vector<Tensor>& v_W_l) {
        
        std::vector<Tensor> ly;
        
        for (size_t k = 0; k < lS_i.size(); k++) {
            // 修复：创建非const版本的张量
            IntTensor sparse_index_group_batch = lS_i[k];  // 复制而不是引用
            
            // 创建当前特征的偏移量张量
            IntTensor sparse_offset_group_batch({lS_o.size(0)});
            for (int64_t i = 0; i < lS_o.size(0); i++) {
                sparse_offset_group_batch[i] = lS_o.data[k * lS_o.size(0) + i];
            }
            
            std::cout << "K device: CPU" << std::endl;
            std::cout << "sparse index shape: [" << sparse_index_group_batch.size(0) << "]" << std::endl;
            std::cout << "sparse offset shape: [" << sparse_offset_group_batch.size(0) << "]" << std::endl;
            
            Tensor per_sample_weights;
            if (k < v_W_l.size() && v_W_l[k].numel > 0) {
                // 创建per_sample_weights（简化版索引选择）
                per_sample_weights = Tensor({sparse_index_group_batch.size(0)});
                for (int64_t i = 0; i < sparse_index_group_batch.size(0); i++) {
                    int64_t idx = sparse_index_group_batch[i];
                    per_sample_weights[i] = v_W_l[k][idx];
                }
            }
            
            // 获取嵌入表
            auto& emb_bag = emb_l[k];
            
            // 创建输出张量
            Tensor V_cuda({sparse_offset_group_batch.size(0), emb_bag->embedding_dim});
            
            if (per_sample_weights.numel > 0) {
                int dst_x = device_map[k % this->ndevices].first;
                int dst_y = device_map[k % this->ndevices].second;
                embeddingbag_forward_cuda(
                    emb_bag->weight.data_ptr(),
                    sparse_index_group_batch.data_ptr(), 
                    sparse_offset_group_batch.data_ptr(),
                    per_sample_weights.data_ptr(),
                    V_cuda.data_ptr(),
                    sparse_offset_group_batch.size(0), 
                    emb_bag->weight.size(0),
                    emb_bag->weight.size(1), 
                    sparse_index_group_batch.size(0),
                    this->src_x, this->src_y, dst_x, dst_y);
            } else {
                int dst_x = device_map[k % this->ndevices].first;
                int dst_y = device_map[k % this->ndevices].second;
                embeddingbag_forward_cuda(
                    emb_bag->weight.data_ptr(),
                    sparse_index_group_batch.data_ptr(), 
                    sparse_offset_group_batch.data_ptr(),
                    nullptr,
                    V_cuda.data_ptr(),
                    sparse_offset_group_batch.size(0), 
                    emb_bag->weight.size(0),
                    emb_bag->weight.size(1), 
                    sparse_index_group_batch.size(0),
                    this->src_x, this->src_y, dst_x, dst_y);
            }
            
            std::cout << "weight size: " << emb_bag->weight.size(0) * emb_bag->weight.size(1) * sizeof(float) << std::endl;
            std::cout << "indices size: " << sparse_index_group_batch.size(0) * sizeof(int64_t) << std::endl;
            std::cout << "offsets size: " << sparse_offset_group_batch.size(0) * sizeof(int64_t) << std::endl;
            std::cout << "V_cuda shape: [" << V_cuda.size(0) << ", " << V_cuda.size(1) << "]" << std::endl;
            
            ly.push_back(V_cuda);
        }
        
        return ly;
    }

    // ... 保持原有的并行前向传播方法，但使用新的张量类型 ...

    Tensor parallel_forward_cuda(const Tensor& dense_x, const IntTensor& lS_o, const std::vector<IntTensor>& lS_i) {
        // 底层MLP (数据并行)
        std::cout << "--------------bot_mlp_forward_cuda start--------------------" << std::endl;
        auto dense_x_chunk = dense_x.chunk(this->ndevices, 0);
        this->ndevices = dense_x_chunk.size();
        std::vector<Tensor> x = mlp_forward_cuda(dense_x_chunk, false);
        std::cout << "--------------bot_mlp_forward_cuda end--------------------" << std::endl;
        
        // 嵌入层
        std::cout << "--------------apply_emb start--------------------" << std::endl;
        std::vector<Tensor> ly = apply_emb(lS_o, lS_i, emb_l, v_W_l);
        std::cout << "--------------apply_emb end--------------------" << std::endl;
        
        // 蝴蝶重组
        std::cout << "--------------interact_features start--------------------" << std::endl;
        std::vector<std::vector<Tensor>> t_list_2d;
        for (size_t k = 0; k < emb_l.size(); k++) {
            std::cout << "ndevices: " << this->ndevices << std::endl;
            std::cout << "ly[k] shape: [" << ly[k].size(0) << ", " << ly[k].size(1) << "]" << std::endl;
            auto ly_chunk = ly[k].chunk(this->ndevices, 0);
            t_list_2d.push_back(ly_chunk);
            std::cout << "scattered shape: [";
            for (const auto& x_t : ly_chunk) {
                std::cout << "[" << x_t.size(0) << ", " << x_t.size(1) << "], ";
            }
            std::cout << "]" << std::endl;
        }
        
        // 调整列表顺序按设备
        std::vector<std::vector<Tensor>> ly_per_device(ndevices);
        for (size_t i = 0; i < ndevices; i++) {
            ly_per_device[i].resize(emb_l.size());
            for (size_t j = 0; j < emb_l.size(); j++) {
                ly_per_device[i][j] = t_list_2d[j][i];
            }
        }
        
        std::vector<Tensor> z;
        for (int64_t k = 0; k < ndevices; k++) {
            std::cout << "x[" << k << "] shape: [" << x[k].size(0) << ", " << x[k].size(1) << "]; ";
            std::cout << "ly_per_device[" << k << "] shape: [" << ly_per_device[k][0].size(0) << ", " << ly_per_device[k][0].size(1) << "]" << std::endl;
            z.push_back(interact_features(x[k], ly_per_device[k]));
        }
        std::cout << "--------------interact_features end--------------------" << std::endl;
        
        // 顶层MLP
        std::cout << "--------------top_mlp_forward_cuda start--------------------" << std::endl;
        std::vector<Tensor> p = mlp_forward_cuda(z, true);
        std::cout << "--------------top_mlp_forward_cuda end--------------------" << std::endl;
        
        Tensor p0 = gather_tensors(p, 0);
        return p0;
    }

    void mlp_forward_parallel(const Tensor& dense_x_chunk, Tensor& output, int device_id, bool is_top_layer = false) {
        const MLP& mlp = is_top_layer ? *top_l : *bot_l;
        std::cout << "device_id: " << device_id << std::endl;

        int dst_x = device_map[device_id].first;
        int dst_y = device_map[device_id].second;
        std::cout << "dst_x: " << dst_x << "; dst_y: " << dst_y << std::endl;
        
        bool is_run = true;
        InterChiplet::sendMessage(dst_x, dst_y, src_x, src_y, &is_run, sizeof(bool));
        int64_t module_id = mlp_comm_id;
        InterChiplet::sendMessage(dst_x, dst_y, src_x, src_y, &module_id, sizeof(int64_t));

        // 从MLP中提取权重和偏置
        int num_layers = mlp.size();
        std::cout << "num_layers: " << num_layers << std::endl;
        
        // 准备参数列表
        int64_t params[4];
        params[0] = dense_x_chunk.size(0); // batch_size
        params[1] = num_layers;      // 层数
        params[2] = dense_x_chunk.size(1); // 输入维度
        
        // 提取每层大小
        std::vector<int> layer_sizes(num_layers + 1);
        std::vector<int> activation_types(num_layers);
        
        layer_sizes[0] = dense_x_chunk.size(1); // 输入维度
        
        for (int i = 0; i < num_layers; i++) {
            const auto& layer = mlp.layers[i];
            layer_sizes[i+1] = layer->weight.size(0); // 输出维度
            activation_types[i] = mlp.activation_types[i];
        }
        
        params[3] = layer_sizes[num_layers]; // 输出维度
        
        // 发送参数到GPU
        InterChiplet::sendMessage(dst_x, dst_y, src_x, src_y, params, 4 * sizeof(int64_t));
        InterChiplet::sendMessage(dst_x, dst_y, src_x, src_y, layer_sizes.data(), (num_layers + 1) * sizeof(int));
        InterChiplet::sendMessage(dst_x, dst_y, src_x, src_y, activation_types.data(), num_layers * sizeof(int));
        
        // 发送权重和偏置
        for (int i = 0; i < num_layers; i++) {
            const auto& layer = mlp.layers[i];
            std::cout << "layer weight shape: [" << layer->weight.size(0) << ", " << layer->weight.size(1) << "]" << std::endl;
            std::cout << "layer bias shape: [" << layer->bias.size(0) << "]" << std::endl;
            
            InterChiplet::sendMessage(dst_x, dst_y, src_x, src_y, 
                                    layer->weight.data_ptr(), 
                                    layer->weight.numel * sizeof(float));
            std::cout << "bias_contiguous shape: " << layer->bias.numel * sizeof(float) << std::endl;
            InterChiplet::sendMessage(dst_x, dst_y, src_x, src_y, 
                                    layer->bias.data_ptr(), 
                                    layer->bias.numel * sizeof(float));
        }
        
        // 修复：创建非const版本用于发送数据
        Tensor dense_x_chunk_copy = dense_x_chunk;  // 复制张量
        InterChiplet::sendMessage(dst_x, dst_y, src_x, src_y, 
                                dense_x_chunk_copy.data_ptr(), 
                                dense_x_chunk_copy.numel * sizeof(float));
        
        // 分配输出张量空间
        output = Tensor({dense_x_chunk.size(0), layer_sizes[num_layers]});
        
        // 接收输出数据
        InterChiplet::receiveMessage(src_x, src_y, dst_x, dst_y, 
                                    output.data_ptr(), 
                                    output.numel * sizeof(float));
    }

    std::vector<Tensor> mlp_forward_cuda(const std::vector<Tensor>& dense_x_chunk_list, bool is_top_layer = false) {
        std::vector<Tensor> output(dense_x_chunk_list.size());
        int divide_num = dense_x_chunk_list.size();
        std::vector<std::thread> threads;
        
        for (int i = 0; i < divide_num; i++) {
            const Tensor& dense_x_chunk = dense_x_chunk_list[i];
            threads.push_back(std::thread(
                std::bind(&DLRMImpl::mlp_forward_parallel, this, 
                         dense_x_chunk, std::ref(output[i]), i, is_top_layer)
            ));
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        return output;
    }

    Tensor sequential_forward(
        const Tensor& dense_x,
        const IntTensor& lS_o,
        const std::vector<IntTensor>& lS_i) {
        
        std::cout << "-----------------------------test 5--------------------------------" << std::endl;
        Tensor x = this->bot_l->forward(dense_x);
        
        std::cout << "x shape: [" << x.size(0) << ", " << x.size(1) << "]" << std::endl;
        std::cout << "-----------------------------test 6--------------------------------" << std::endl;
        std::vector<Tensor> ly = apply_emb(lS_o, lS_i, this->emb_l, this->v_W_l);
        std::cout << "-----------------------------test 7--------------------------------" << std::endl;
        std::cout << "ly size: " << ly.size() << std::endl;
        std::cout << "ly[0] shape: [" << ly[0].size(0) << ", " << ly[0].size(1) << "]" << std::endl;
        Tensor z = interact_features(x, ly);
        std::cout << "-----------------------------test 8--------------------------------" << std::endl;
        Tensor p = this->top_l->forward(z);
        std::cout << "-----------------------------test 9--------------------------------" << std::endl;
        return p;
    }

    Tensor forward(const Tensor& dense_x,
        const IntTensor& lS_o,
        const std::vector<IntTensor>& lS_i) {
        
        if (this->ndevices > 0) {
            return parallel_forward_cuda(dense_x, lS_o, lS_i);
        } else {
            return sequential_forward(dense_x, lS_o, lS_i);
        }
    }
    
    // 构造函数
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
        const std::string& loss_function,
        int src_x,
        int src_y,
        std::unordered_map<int, std::pair<int, int>> device_map) {
        
        if (m_spa != 0 && !ln_emb.empty() && !ln_bot.empty() && 
            !ln_top.empty() && !arch_interaction_op.empty()) {
            
            // 保存参数
            this->ndevices = ndevices;
            this->output_d = 0;
            // ... 继续构造函数 ...
            this->parallel_model_batch_size = -1;
            this->parallel_model_is_not_prepared = true;
            this->arch_interaction_op = arch_interaction_op;
            this->arch_interaction_itself = arch_interaction_itself;
            this->sync_dense_params = sync_dense_params;
            this->loss_threshold = loss_threshold;
            this->loss_function = loss_function;
            this->device_map = device_map;
            
            if (weighted_pooling != "" && weighted_pooling != "fixed") {
                this->weighted_pooling = "learned";
            } else {
                this->weighted_pooling = weighted_pooling;
            }
            
            // 创建QR嵌入相关变量
            this->qr_flag = qr_flag;
            if (this->qr_flag) {
                this->qr_collisions = qr_collisions;
                this->qr_operation = qr_operation;
                this->qr_threshold = qr_threshold;
            }
            
            // 创建MD嵌入相关变量
            this->md_flag = md_flag;
            if (this->md_flag) {
                this->md_threshold = md_threshold;
            }
            
            // 创建操作符 - 无论ndevices如何都初始化嵌入层
            auto emb_result = this->create_emb(m_spa, ln_emb, weighted_pooling);
            this->emb_l = std::move(emb_result.first);
            this->v_W_l = std::move(emb_result.second);
            
            std::cout << "emb_l size: " << this->emb_l.size() << std::endl;
            
            std::cout << "-----------------------------test 3--------------------------------" << std::endl;
            this->bot_l = this->create_mlp(ln_bot, sigmoid_bot);
            this->top_l = this->create_mlp(ln_top, sigmoid_top);
            std::cout << "-----------------------------test 4--------------------------------" << std::endl;
            
            // 量化
            this->quantize_emb = false;
            this->quantize_bits = 32;

            // 保存新参数
            this->src_x = src_x;
            this->src_y = src_y;
        }
    }
};

// DLRM包装类
class DLRM {
private:
    std::unique_ptr<DLRMImpl> impl;
    
public:
    DLRM(int64_t m_spa,
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
         const std::string& loss_function,
         int src_x,
         int src_y,
         std::unordered_map<int, std::pair<int, int>> device_map) {
        
        impl = std::make_unique<DLRMImpl>(
            m_spa, ln_emb, ln_bot, ln_top, arch_interaction_op, arch_interaction_itself,
            sigmoid_bot, sigmoid_top, sync_dense_params, loss_threshold, ndevices,
            qr_flag, qr_operation, qr_collisions, qr_threshold, md_flag, md_threshold,
            weighted_pooling, loss_function, src_x, src_y, device_map
        );
    }
    
    Tensor forward(const Tensor& dense_x, const IntTensor& lS_o, const std::vector<IntTensor>& lS_i) {
        return impl->forward(dense_x, lS_o, lS_i);
    }
    
    DLRMImpl* operator->() { return impl.get(); }
    const DLRMImpl* operator->() const { return impl.get(); }
};

bool readCSV(std::string csv_path, std::unordered_map<int, std::pair<int, int>> &device_map, int64_t &device_num, int &cpu_node, int topology_width) {
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

int main(int argc, char* argv[]) {
    int src_x = atoi(argv[1]);
    int src_y = atoi(argv[2]);

    // 模型参数
    std::cout << "----------------------start--------------------------------" << std::endl;
    std::vector<int64_t> ln_emb = {30, 20, 10};  // 嵌入表大小
    std::vector<int64_t> ln_bot = {40, 30, 50};  // 底层MLP结构
    int64_t num_fea = ln_emb.size() + 1;
    int64_t m_den_out = ln_bot[ln_bot.size() - 1]; 
    int64_t num_int = (num_fea * (num_fea - 1)) / 2 + m_den_out;
    std::vector<int64_t> ln_top = {num_int, 4, 2, 1};
    int64_t mini_batch_size = 10;            // 样本数量
    int64_t num_indices_per_lookup = 2;      // 每个样本每个特征的索引数量
    bool fixed_indices = false;              // 是否使用固定数量的索引
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
    std::unordered_map<int, std::pair<int, int>> device_map;
    
    std::cout << "---------------test 1--------------------------------" << std::endl;
    int64_t device_num = 0;
    int cpu_node = -1;
    int topology_width = 2;
    try {
        if (!readCSV("../../mapDevice.csv", device_map, device_num, cpu_node, topology_width)) {
            std::cerr << "读取CSV文件失败" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "读取CSV文件失败: " << e.what() << std::endl;
        return 1;
    }
    int64_t ndevices = std::min({device_num, mini_batch_size, num_fea - 1});
    
    std::cout << "---------------test 2--------------------------------" << std::endl;
    
    // 1. 生成密集特征 X: [mini_batch_size, ln_bot[0]]
    Tensor dense_x = RandomGenerator::rand({mini_batch_size, ln_bot[0]});
    
    std::cout << "dense_x shape: [" << dense_x.size(0) << ", " << dense_x.size(1) << "]" << std::endl;
    std::cout << "dense_x first few values: ";
    for (int i = 0; i < std::min(10L, dense_x.numel); i++) {
        std::cout << dense_x[i] << " ";
    }
    std::cout << std::endl << std::endl;
    
    // 2. 生成稀疏特征偏移量和索引值
    std::vector<int64_t> offsets_data;
    std::vector<std::vector<int64_t>> indices_data(ln_emb.size());
    
    // 为每个样本确定特征值数量
    std::vector<std::vector<int64_t>> indices_per_sample(ln_emb.size(), std::vector<int64_t>(mini_batch_size));
    
    for (size_t f = 0; f < ln_emb.size(); f++) {
        for (int64_t i = 0; i < mini_batch_size; i++) {
            if (fixed_indices) {
                indices_per_sample[f][i] = num_indices_per_lookup;
            } else {
                indices_per_sample[f][i] = RandomGenerator::randint(1, num_indices_per_lookup + 1, {1})[0];
            }
        }
    }
    
    // 计算每个特征的累积偏移量
    std::vector<std::vector<int64_t>> offsets_per_feature(ln_emb.size(), std::vector<int64_t>(mini_batch_size + 1, 0));
    
    for (size_t f = 0; f < ln_emb.size(); f++) {
        for (int64_t i = 0; i < mini_batch_size; i++) {
            offsets_per_feature[f][i + 1] = offsets_per_feature[f][i] + indices_per_sample[f][i];
        }
        
        // 生成随机索引
        indices_data[f].resize(offsets_per_feature[f][mini_batch_size]);
        for (int64_t i = 0; i < offsets_per_feature[f][mini_batch_size]; i++) {
            indices_data[f][i] = RandomGenerator::randint(0, ln_emb[f], {1})[0];
        }
    }
    
    // 创建lS_o张量 (注意这里是整个张量，不是列表)，lS_o是稀疏特征的偏移量
    IntTensor lS_o({static_cast<int64_t>(ln_emb.size()), mini_batch_size + 1});
    for (size_t f = 0; f < ln_emb.size(); f++) {
        for (int64_t i = 0; i <= mini_batch_size; i++) {
            lS_o.data[f * (mini_batch_size + 1) + i] = offsets_per_feature[f][i];
        }
    }
    
    // 创建lS_i列表（注意这是一个包含张量的向量）
    std::vector<IntTensor> lS_i;
    for (size_t f = 0; f < ln_emb.size(); f++) {
        IntTensor indices_tensor({static_cast<int64_t>(indices_data[f].size())}, indices_data[f]);
        lS_i.push_back(indices_tensor);
    }
    
    // 由于lS_o的形状与示例不完全匹配（示例是[3, 10]而不是[3, 11]），我们修剪最后一列
    IntTensor lS_o_trimmed({static_cast<int64_t>(ln_emb.size()), mini_batch_size});
    for (size_t f = 0; f < ln_emb.size(); f++) {
        for (int64_t i = 0; i < mini_batch_size; i++) {
            lS_o_trimmed.data[f * mini_batch_size + i] = lS_o.data[f * (mini_batch_size + 1) + i];
        }
    }
    
    // 3. 打印生成的数据
    std::cout << "lS_o shape: [" << lS_o_trimmed.size(0) << ", " << lS_o_trimmed.size(1) << "]" << std::endl;
    std::cout << "lS_o first few values: ";
    for (int i = 0; i < std::min(10L, lS_o_trimmed.numel); i++) {
        std::cout << lS_o_trimmed[i] << " ";
    }
    std::cout << std::endl << std::endl;
    
    std::cout << "lS_i (list of tensors):" << std::endl;
    for (size_t f = 0; f < lS_i.size(); f++) {
        std::cout << "  - Feature " << f << " shape: [" << lS_i[f].size(0) << "]" << std::endl;
        std::cout << "    First few values: ";
        for (int i = 0; i < std::min(5L, lS_i[f].numel); i++) {
            std::cout << lS_i[f][i] << " ";
        }
        std::cout << std::endl;
    }
    
    SimRoiStart();
    auto dlrm_model = DLRM(m_den_out, ln_emb, ln_bot, ln_top, arch_interaction_op, arch_interaction_itself, 
                          sigmoid_bot, sigmoid_top, sync_dense_params, loss_threshold, ndevices, qr_flag, 
                          qr_operation, qr_collisions, qr_threshold, md_flag, md_threshold, weighted_pooling, 
                          loss_function, src_x, src_y, device_map);
    
    Tensor result = dlrm_model.forward(dense_x, lS_o_trimmed, lS_i);
    std::cout << "result shape: [" << result.size(0) << ", " << result.size(1) << "]" << std::endl;
    std::cout << "result first few values: ";
    for (int i = 0; i < std::min(10L, result.numel); i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    
    bool is_run = false;
    for (int i = 0; i < device_num; i++) {
        int dstX = device_map[i].first;
        int dstY = device_map[i].second;
        InterChiplet::sendMessage(dstX, dstY, src_x, src_y, &is_run, sizeof(bool));
    }
    SimRoiEnd();
    
    return 0;
}