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
    int size = 12*12*8;
    int64_t* size_A = new int64_t[size];
    std::string fileName = InterChiplet::receiveSync(3, 3, idX, idY);
    global_pipe_comm.read_data(fileName.c_str(), size_A, size * sizeof(int64_t));
    long long int time_end = InterChiplet::readSync(timeNow, 3, 3, idX, idY, size * sizeof(int64_t), 0);

    system("cd $SIMULATOR_ROOT/scale-sim-v2 && pip3 install -e . && python -m scalesim.scale -c $SIMULATOR_ROOT/scale-sim-v2/configs/google.cfg -t $SIMULATOR_ROOT/scale-sim-v2/topologies/dlrm/dlrm_fwd.csv -p $SIMULATOR_ROOT/scale-sim-v2/test_runs");

    std::ifstream inputFile("$SIMULATOR_ROOT/scale-sim-v2/test_runs/total_cycle.txt");
    float time;
    inputFile >> time;
    long long unsigned int true_time = (long long unsigned int)time;
    // std::cout<< "Time taken: " << true_time << std::endl;
    timeNow = true_time + time_end; 

    int64_t* ans = new int64_t[112*112*128];
    for(int i=0;i<112*112*128;i++){
        ans[i] = i;
    }

    fileName = InterChiplet::sendSync(idX, idY, 3, 3);
    global_pipe_comm.write_data(fileName.c_str(), ans, size * sizeof(int64_t));
    time_end = InterChiplet::writeSync(time_end, idX, idY, 3, 3, size * sizeof(int64_t), 0);

    delete[] size_A;
    delete[] ans;
    
}
