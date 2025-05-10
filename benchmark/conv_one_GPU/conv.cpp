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
    int width = 1024;
    int height = 14;
    int in_channels = 14;
    int out_channels = 7;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    int out_width = (width - kernel_size + 2 * padding) / stride + 1;
    int out_height = (height - kernel_size + 2 * padding) / stride + 1;

    int size = width * height * in_channels;
    int size_out = out_width * out_height * out_channels;
    float *M = new float[size];
    float *M_out = new float[size_out];
    for (int i = 0; i < size; i++) {
        M[i] = (rand() % 255) * 1e4;
    }

    for(int i=0; i < 1;i++){
        InterChiplet::sendMessage(0, i, idX, idY, M, size * sizeof(float));
    }
    for (int i = 0; i < 1; i++) {
        InterChiplet::receiveMessage(idX, idY, 0, i, M_out, size_out * sizeof(float));
    }

    delete[] M;
    delete[] M_out;
    return 0;
}