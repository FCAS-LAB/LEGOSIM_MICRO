#include <fstream>
#include <iostream>
#include <vector>
#include "apis_c.h"

int Row = 100;
int Col = 100;

void random_init(int64_t * data, int size){
    int i;
    for (i = 0; i < size; i++){
        //data[i] = rand();
        data[i] = 128;
    }
}
// 1000*1000 和 1000*1000 相乘
int idX, idY;
using namespace std;
int main(int argc, char **argv) {
    idX = atoi(argv[1]);
    idY = atoi(argv[2]);

    int size = 1e4;
    int64_t *test = new int64_t[size];
    int64_t *test_ans = new int64_t[size];
    random_init(test, size);
    for(int i=0; i<10; i++){
        InterChiplet::sendMessage(0, 0, idX, idY, test, size*sizeof(int64_t));
        InterChiplet::receiveMessage(idX, idY, 0, 0, test_ans, size*sizeof(int64_t));
    }
    for(int i=0; i<10; i++){
        InterChiplet::sendMessage(0, 1, idX, idY, test, size*sizeof(int64_t));
        InterChiplet::receiveMessage(idX, idY, 0, 1, test_ans, size*sizeof(int64_t));
    }
    for(int i=0; i<10; i++){
        InterChiplet::sendMessage(1, 0, idX, idY, test, size*sizeof(int64_t));
        InterChiplet::receiveMessage(idX, idY, 1, 0, test_ans, size*sizeof(int64_t));
    }
    for(int i=0; i<10; i++){
        InterChiplet::sendMessage(2, 0, idX, idY, test, size*sizeof(int64_t));
        InterChiplet::receiveMessage(idX, idY, 2, 0, test_ans, size*sizeof(int64_t));
    }
    
    delete[] test;
    delete[] test_ans;
    
}
