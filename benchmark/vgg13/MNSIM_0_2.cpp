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
    uint64_t* interdata = new uint64_t[112*112*64/8];
    InterChiplet::SyncProtocol::pipeSync(0, 1, idX, idY);

    char * fileName = InterChiplet::SyncProtocol::pipeName(0, 1, idX, idY);
    global_pipe_comm.read_data(fileName, interdata, 112*112*64);
    delete fileName;
    long long int time_end = InterChiplet::SyncProtocol::readSync(timeNow, 0, 1, idX, idY, 112*112*64, 0);

    system("cd /home/qc/Chiplet_Heterogeneous_newVersion_gem5/Chiplet_Heterogeneous_newVersion/MNSIMChiplet;python3 MNSIM_Chiplet.py -ID1 0 -ID2 2");

    std::ifstream inputFile("/home/qc/Chiplet_Heterogeneous_newVersion_gem5/Chiplet_Heterogeneous_newVersion/MNSIMChiplet/result_0_2.res");
    float time;
    inputFile >> time;
    long long unsigned int true_time = (long long unsigned int)time;
    timeNow = true_time + time_end; 

    uint64_t* interdata2 = new uint64_t[112*112*128/8];

    InterChiplet::SyncProtocol::pipeSync(idX, idY, 0, 3);

    fileName = InterChiplet::SyncProtocol::pipeName(idX, idY, 0, 3);
    global_pipe_comm.write_data(fileName, interdata2, 112*112*128);
    delete fileName;

    InterChiplet::SyncProtocol::writeSync(timeNow, idX, idY, 0, 3, 112*112*128, 0);
}
