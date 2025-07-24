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
        std::string fileName = InterChiplet::receiveSync(2, 2, idX, idY);
        global_pipe_comm.read_data(fileName.c_str(), size_A, size * sizeof(int64_t));
        long long int time_end = InterChiplet::readSync(timeNow, 2, 2, idX, idY, size * sizeof(int64_t), 0);

        long long unsigned int true_time = 50000;
        timeNow = true_time + time_end; 

        int64_t* ans = new int64_t[size];
        for(int i=0;i<size;i++){
            ans[i] = i;
        }

        fileName = InterChiplet::sendSync(idX, idY, 2, 2);
        global_pipe_comm.write_data(fileName.c_str(), ans, size * sizeof(int64_t));
        time_end = InterChiplet::writeSync(timeNow, idX, idY, 2, 2, size * sizeof(int64_t), 0);
        delete[] ans;
    }
    

    delete[] size_A;  
}