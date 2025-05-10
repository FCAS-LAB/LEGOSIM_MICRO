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
    uint64_t* interdata = new uint64_t[56*56*64/8];

    std::string fileName = InterChiplet::receiveSync(0, 0, idX, idY);
    global_pipe_comm.read_data(fileName.c_str(), interdata, 56*56*64);

    long long int time_end = InterChiplet::readSync(timeNow, 0, 0, idX, idY, 56*56*64, 0);

    system("cd /home/gh/Chiplet_Heterogeneous_newVersion/MNSIMChiplet;python3 MNSIM_Chiplet.py -ID1 1 -ID2 0");

    std::ifstream inputFile("/home/gh/Chiplet_Heterogeneous_newVersion/MNSIMChiplet/result_1_0.res");
    float time;
    inputFile >> time;
    long long unsigned int true_time = (long long unsigned int)time;
    timeNow = true_time + time_end; 

    uint64_t* interdata2 = new uint64_t[28*28*128/8];


    fileName = InterChiplet::sendSync(idX, idY, 2, 0);
    global_pipe_comm.write_data(fileName.c_str(), interdata2, 28*28*128);


    InterChiplet::writeSync(timeNow, idX, idY, 2, 0, 28*28*128, 0);
}
