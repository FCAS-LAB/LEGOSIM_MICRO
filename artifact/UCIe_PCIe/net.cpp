#include <fstream>
#include <iostream>
#include <vector>
#include "apis_c.h"
using namespace std;

int main(int argc, char **argv) {
    int idX = atoi(argv[1]);
    int idY = atoi(argv[2]);

    int size = 1e4;
    int64_t *test = new int64_t[size];
    int64_t *test_ans = new int64_t[size];
    
    for (int i=0; i<10; i++){
        InterChiplet::sendMessage(0, 0, idX, idY, test, size*sizeof(int64_t));
        InterChiplet::receiveMessage(idX, idY, 0, 0, test_ans, size*sizeof(int64_t));
        InterChiplet::sendMessage(0, 1, idX, idY, test, size*sizeof(int64_t));
        InterChiplet::receiveMessage(idX, idY, 0, 1, test_ans, size*sizeof(int64_t));
        InterChiplet::sendMessage(0, 2, idX, idY, test, size*sizeof(int64_t));
        InterChiplet::receiveMessage(idX, idY, 0, 2, test_ans, size*sizeof(int64_t));
        InterChiplet::sendMessage(0, 3, idX, idY, test, size*sizeof(int64_t));
        InterChiplet::receiveMessage(idX, idY, 0, 3, test_ans, size*sizeof(int64_t));
        InterChiplet::sendMessage(1, 0, idX, idY, test, size*sizeof(int64_t));
        InterChiplet::receiveMessage(idX, idY, 1, 0, test_ans, size*sizeof(int64_t));
        InterChiplet::sendMessage(1, 1, idX, idY, test, size*sizeof(int64_t));
        InterChiplet::receiveMessage(idX, idY, 1, 1, test_ans, size*sizeof(int64_t));
        InterChiplet::sendMessage(1, 2, idX, idY, test, size*sizeof(int64_t));
        InterChiplet::receiveMessage(idX, idY, 1, 2, test_ans, size*sizeof(int64_t));
        InterChiplet::sendMessage(1, 3, idX, idY, test, size*sizeof(int64_t));
        InterChiplet::receiveMessage(idX, idY, 1, 3, test_ans, size*sizeof(int64_t));
    }   
}