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
    int size = 1e4;
    int64_t* size_A = new int64_t[size];
    for(int i=0; i<10; i++){
        std::string fileName = InterChiplet::receiveSync(5, 5, idX, idY);
        global_pipe_comm.read_data(fileName.c_str(), size_A, size * sizeof(int64_t));
        long long int time_end = InterChiplet::readSync(timeNow, 5, 5, idX, idY, size * sizeof(int64_t), 0);

        // system("cd /home/qc/Chiplet_Heterogeneous_newVersion_gem5/Chiplet_Heterogeneous_newVersion/MNSIMChiplet;python3 MNSIM_Chiplet.py -ID1 0 -ID2 2");

        std::ifstream inputFile("/home/qc/scale-sim-v2/test_runs/GoogleTPU_v1_ws/total_cycle.txt");
        float time;
        inputFile >> time;
        long long unsigned int true_time = (long long unsigned int)time;
        timeNow = true_time + time_end; 

        int64_t* ans = new int64_t[size];
        // for(int i=0;i<112*112*128;i++){
        //     ans[i] = i;
        // }

        fileName = InterChiplet::sendSync(idX, idY, 5, 5);
        global_pipe_comm.write_data(fileName.c_str(), ans, size * sizeof(int64_t));
        time_end = InterChiplet::writeSync(time_end, idX, idY, 5, 5, size * sizeof(int64_t), 0);
        delete[] ans;
    }
    

    delete[] size_A;
    
    
}
