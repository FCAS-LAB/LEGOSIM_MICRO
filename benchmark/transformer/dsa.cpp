#include <fstream>
#include <iostream>
#include <stdlib.h>
#include "apis_c.h"
#include "../../interchiplet/includes/pipe_comm.h"

int idX, idY;

InterChiplet::PipeComm global_pipe_comm;

int main(int argc, char** argv)
{
    idX = atoi(argv[1]);
    idY = atoi(argv[2]);
    long long unsigned int timeNow = 1;
    int size_input = 1;
    int size = 10*10*10*10;
    int64_t* size_A = new int64_t;
    std::string fileName = InterChiplet::receiveSync(5, 5, idX, idY);
    global_pipe_comm.read_data(fileName.c_str(), size_A, size_input * sizeof(int64_t));
    long long int time_end = InterChiplet::readSync(timeNow, 5, 5, idX, idY, size_input * sizeof(int64_t), 0);

    long long unsigned int true_time = 5000;
    timeNow = true_time + time_end; 

    float* ans = new float[size];
    for(int i=0;i<size;i++){
        ans[i] = i;
    }

    fileName = InterChiplet::sendSync(idX, idY, 5, 5);
    global_pipe_comm.write_data(fileName.c_str(), ans, size * sizeof(float));
    time_end = InterChiplet::writeSync(timeNow, idX, idY, 5, 5, size * sizeof(float), 0);

    timeNow = true_time + time_end; 
    int size_2 = 10*10*10*5;
    float* ans_2 = new float[size_2];
    for(int i=0;i<size_2;i++){
        ans_2[i] = i;
    }

    fileName = InterChiplet::sendSync(idX, idY, 5, 5);
    global_pipe_comm.write_data(fileName.c_str(), ans_2, size_2 * sizeof(float));
    time_end = InterChiplet::writeSync(timeNow, idX, idY, 5, 5, size_2 * sizeof(float), 0);

    delete size_A;
    delete[] ans;
    delete[] ans_2;
}