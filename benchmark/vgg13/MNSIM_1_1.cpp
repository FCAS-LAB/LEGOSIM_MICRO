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
    uint64_t* interdata = new uint64_t[56*56*256/8];
    InterChiplet::SyncProtocol::pipeSync(1, 0, idX, idY);

    char * fileName = InterChiplet::SyncProtocol::pipeName(1, 0, idX, idY);
    global_pipe_comm.read_data(fileName, interdata, 56*56*256);
    delete fileName;
    long long int time_end = InterChiplet::SyncProtocol::readSync(timeNow, 1, 0, idX, idY, 56*56*256, 0);

    system("cd /home/zzl/ws2/interface/Chiplet_Heterogeneous_newVersion/MNSIMChiplet;python3 MNSIM_Chiplet.py -ID1 1 -ID2 1");

    std::ifstream inputFile("/home/zzl/ws2/interface/Chiplet_Heterogeneous_newVersion/MNSIMChiplet/result_1_1.res");
    float time;
    inputFile >> time;
    long long unsigned int true_time = (long long unsigned int)time;
    timeNow = true_time + time_end; 

    uint64_t* interdata2 = new uint64_t[28*28*256/8];

    InterChiplet::SyncProtocol::pipeSync(idX, idY, 1, 2);

    fileName = InterChiplet::SyncProtocol::pipeName(idX, idY, 1, 2);
    global_pipe_comm.write_data(fileName, interdata2, 28*28*256);
    delete fileName;

    InterChiplet::SyncProtocol::writeSync(timeNow, idX, idY, 1, 2, 28*28*256, 0);
}
