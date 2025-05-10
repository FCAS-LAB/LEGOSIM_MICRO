#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>
#include <cassert>
#include <cstdio>
#include <stdlib.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <string>
#include "apis_c.h"
#include "../../interchiplet/includes/pipe_comm.h"
InterChiplet::PipeComm global_pipe_comm;

#include "apis_c.h"


int Row_A = 20;
int Col_A = 100;
int Row_B = 100;
int Col_B = 100;



void matrix_mul_cpu(const int64_t* M, const int64_t* N, int64_t* P, int height_M, int width_M, int width_N) {
    for (int i = 0; i < height_M; ++i) {
        for(int j=0;j<width_N;j++){
            P[i * width_N + j] = 0;
            for (int k = 0; k < width_M; ++k) {
                P[i * width_N + j] += M[i * width_M + k] * N[k * width_N + j];
            }
        } 
    }
}

int main(int argc, char** argv) {
    // 读取本进程所代表的chiplet编号
    int idX = atoi(argv[1]);
    int idY = atoi(argv[2]);
    // if(idY == 3){
    //     Row_A = 1;
    // }
    int64_t* dataA = new int64_t[Row_A * Col_A];
    int64_t* dataB = new int64_t[Row_B * Col_B];
    int64_t* dataC = new int64_t[Row_A * Col_B];

    // InterChiplet::receiveMessage(idX, idY, 3, 3, dataA, sizeof(int64_t) * Row_A * Col_A);
    long long unsigned int timeNow = 1;
    std::string fileName = InterChiplet::receiveSync(3, 3, idX, idY);
    global_pipe_comm.read_data(fileName.c_str(), dataA, Row_A * Col_A * sizeof(int64_t));
    long long int time_end = InterChiplet::readSync(timeNow, 3, 3, idX, idY, Row_A * Col_A * sizeof(int64_t), 0);
    // InterChiplet::receiveMessage(idX, idY, 3, 3, dataB, sizeof(int64_t) * Row_B * Col_B);
    fileName = InterChiplet::receiveSync(3, 3, idX, idY);
    global_pipe_comm.read_data(fileName.c_str(), dataB, Row_B * Col_B * sizeof(int64_t));
    time_end = InterChiplet::readSync(time_end, 3, 3, idX, idY, Row_B * Col_B * sizeof(int64_t), 0);

    // calculate
    std::cout<<"--------------------------------------开始计算--------------------------------------"<<std::endl;
    matrix_mul_cpu(dataA, dataB, dataC, Row_A,Col_A,Col_B);
    std::cout<<"--------------------------------------计算完成--------------------------------------"<<std::endl;
    // InterChiplet::sendMessage(3, 3, idX, idY, dataC, Row_A * Col_B * sizeof(int64_t));
    fileName = InterChiplet::sendSync(idX, idY, 3, 3);
    global_pipe_comm.write_data(fileName.c_str(), dataC, Row_A * Col_B * sizeof(int64_t));
    time_end = InterChiplet::writeSync(time_end, idX, idY, 3, 3, Row_A * Col_B * sizeof(int64_t), 0);

    delete[] dataA;
    delete[] dataB;
    delete[] dataC;

    return 0;
}
