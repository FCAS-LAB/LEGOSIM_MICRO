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
    uint64_t* interdata = new uint64_t[224*224*3/8];

    InterChiplet::receiveMessage(idX, idY, 2, 2, interdata, 224*224*3);

    system("cd /home/zzl/ws2/interface/Chiplet_Heterogeneous_newVersion/MNSIMChiplet;python3 MNSIM_Chiplet.py -ID1 0 -ID2 0");

    //std::ifstream inputFile("/home/zzl/ws2/interface/Chiplet_Heterogeneous_newVersion/MNSIMChiplet/result_0_0.res");
    //float time;
    //inputFile >> time;
    //long long unsigned int true_time = (long long unsigned int)time;
    //timeNow = true_time + time_end; 

    uint64_t* interdata2 = new uint64_t[224*224*64/8];

    InterChiplet::sendMessage(0, 1, idX, idY, interdata2, 224*224*64);
}
