#include <fstream>
#include <iostream>
#include "apis_c.h"
#include<random>
#include <ctime>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

void random_init(int64_t * data, int size){
    int i;
    for (i = 0; i < size; i++){
        //data[i] = rand();
        data[i] = 128;
    }
}
int main(int argc, char **argv){
    int idX = atoi(argv[1]);
    int idY = atoi(argv[2]);
    int Row_A = 20;
    int Col_A = 100;
    int Row_B = 100;
    int Col_B = 100;
    int64_t* dataA = new int64_t[Row_A * Col_A];
    int64_t* dataB = new int64_t[Row_B * Col_B];
    int64_t* dataC = new int64_t[Row_A * Col_B];
    std::srand(std::time(nullptr));
    for (int i = 0; i < Row_A * Col_A; i++) {
        dataA[i] = std::rand() % 100;
    }
    for (int i = 0; i < Row_B * Col_B; i++) {
        dataB[i] = std::rand() % 100;
    }

    InterChiplet::sendMessage(0, 5, idX, idY, dataA, Row_A * Col_A * sizeof(int64_t));
    InterChiplet::sendMessage(0, 5, idX, idY, dataB, Row_B * Col_B * sizeof(int64_t));
    InterChiplet::receiveMessage(idX, idY, 0, 5, dataC, Row_A * Col_B * sizeof(int64_t));
    delete[] dataA;
    delete[] dataB;
    delete[] dataC;
    std::cout<<"--------------------------------------npu over--------------------------------------"<<std::endl;
    int size = 12*12*8;
    int64_t *test = new int64_t[size];
    int64_t *test_ans = new int64_t[size];
    random_init(test, size);
    InterChiplet::sendMessage(5, 0, idX, idY, test, size*sizeof(int64_t));
    InterChiplet::receiveMessage(idX, idY, 5, 0, test_ans, size*sizeof(int64_t));
    delete[] test;
    delete[] test_ans;
    std::cout<<"-------------------------------------mnsim over-------------------------------------"<<std::endl;
    int width = 64;
    int height = 64;
    int in_channels = 3;
    int out_channels = 64;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    int out_width = (width - kernel_size + 2 * padding) / stride + 1;
    int out_height = (height - kernel_size + 2 * padding) / stride + 1;

    size = width * height * in_channels;
    int size_out = out_width * out_height * out_channels;
    float *M = new float[size];
    float *M_out = new float[size_out];
    for (int i = 0; i < size; i++) {
        M[i] = (rand() % 255) * 1e4;
    }

    for(int i=0; i<4;i++){
        InterChiplet::sendMessage(0, i, idX, idY, M, size * sizeof(float));
    }
    for (int i = 0; i < 4; i++) {
        InterChiplet::receiveMessage(idX, idY, 0, i, M_out, size_out * sizeof(float));
    }
    // for(int i = 0; i < size_out; i++){
    //     std::cout << M_out[i] << " ";
    // }

    delete[] M;
    delete[] M_out;
    return 0;
}