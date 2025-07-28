#include <fstream>
#include <iostream>
#include <vector>
#include "apis_c.h"


void random_init(int64_t * data, int size){
    int i;
    for (i = 0; i < size; i++){
        //data[i] = rand();
        data[i] = 128;
    }
}

int idX, idY;
using namespace std;
int main(int argc, char **argv) {
    idX = atoi(argv[1]);
    idY = atoi(argv[2]);

    int size = 12*12*8;
    int64_t *test = new int64_t[size];
    int64_t *test_ans = new int64_t[size];
    random_init(test, size);
    InterChiplet::sendMessage(3, 0, idX, idY, test, size*sizeof(int64_t));
    InterChiplet::receiveMessage(idX, idY, 3, 0, test_ans, size*sizeof(int64_t));
    delete[] test;
    delete[] test_ans;
    std::cout<<"-------------------------------------NPU over-------------------------------------"<<std::endl;
}
