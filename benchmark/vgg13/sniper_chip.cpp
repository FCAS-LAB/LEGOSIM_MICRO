#include <fstream>
#include <iostream>
#include "apis_c.h"
#include "/usr/local/include/onnxruntime/core/session/onnxruntime_cxx_api.h"

int idX,idY;

int main(int argc, char** argv)
{
    idX = atoi(argv[1]);
    idY = atoi(argv[2]);

    int64_t *A = (int64_t *)malloc((sizeof(int64_t)/8) * 224 * 224 * 3);
    int64_t *B = (int64_t *)malloc((sizeof(int64_t)/8) * 7 * 7 * 512);

    for (int i = 0; i < (224 * 224 * 3)/8; i++) {
        A[i] = rand() % 51;
    }
    
    InterChiplet::sendMessage(0, 0, idX, idY, A, (sizeof(int64_t)/8) * 224 * 224 * 3);
    InterChiplet::receiveMessage(idX, idY, 2, 1, B, (sizeof(int64_t)/8) * 7 * 7 * 512);

    // std::cout << "111" << std::endl;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        // Load ONNX model
    Ort::SessionOptions session_options;
    Ort::Session session(env, "/home/zzl/ws2/interface/Chiplet_Heterogeneous_newVersion/benchmark/vgg13/vgg13.onnx", session_options);
    // std::cout << "222" << std::endl;
    std::vector<float> input_data(512 * 7 * 7 , 0.5);

    std::vector<int64_t> tensorShape = {1,512, 7, 7};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
    input_data.data(), 1*512*7*7 , tensorShape.data() , 4); 
    // std::cout << "333" << std::endl;
    const char* input_names[] = {"input"};  
    const char* output_names[] = {"output"};
    auto output_tensors = session.Run(
    Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    // std::cout << "444" << std::endl;
    float* floatarr = output_tensors[0].GetTensorMutableData<float>();

    std::cout << "推理结果：" << *floatarr << std::endl;
}