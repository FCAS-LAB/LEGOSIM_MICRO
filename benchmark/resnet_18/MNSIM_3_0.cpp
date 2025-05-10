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
    uint64_t* interdata = new uint64_t[14*14*256/8];

    std::string fileName = InterChiplet::receiveSync(2, 0, idX, idY);
    global_pipe_comm.read_data(fileName.c_str(), interdata, 14*14*256);

    long long int time_end = InterChiplet::readSync(timeNow, 2, 0, idX, idY, 14*14*256, 0);

    system("cd /home/gh/Chiplet_Heterogeneous_newVersion/MNSIMChiplet;python3 MNSIM_Chiplet.py -ID1 3 -ID2 0");

    std::ifstream inputFile("/home/gh/Chiplet_Heterogeneous_newVersion/MNSIMChiplet/result_3_0.res");
    float time;
    inputFile >> time;
    long long unsigned int true_time = (long long unsigned int)time;
    timeNow = true_time + time_end; 

    uint64_t* interdata2 = new uint64_t[7*7*512/8];


    fileName = InterChiplet::sendSync(idX, idY, 4, 0);
    global_pipe_comm.write_data(fileName.c_str(), interdata2, 7*7*512);


    InterChiplet::writeSync(timeNow, idX, idY, 4, 0, 7*7*512, 0);
}
