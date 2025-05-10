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

    int size = 12*12*8;
    int64_t *test = new int64_t[size];
    int64_t *test_ans = new int64_t[size];
    random_init(test, size);
    InterChiplet::sendMessage(3, 0, idX, idY, test, size*sizeof(int64_t));
    InterChiplet::receiveMessage(idX, idY, 3, 0, test_ans, size*sizeof(int64_t));
    delete[] test;
    delete[] test_ans;
    std::cout<<"-------------------------------------mnsim over-------------------------------------"<<std::endl;
    vector<int64_t*> A,A1;
    for(int i = 0; i < 5; i++) {
        int64_t *a = new int64_t[Row/5 * Col];
        for (int i = 0; i < Row/5 * Col; i++) {
            a[i] = rand() % 51;
        }
        A.push_back(a);
    }
    for(int i = 0; i < 5; i++) {
        int64_t *a = new int64_t[Row/5 * Col*1000];
        for (int i = 0; i < Row/5 * Col*1000; i++) {
            a[i] = rand() % 51;
        }
        A1.push_back(a);
    }

    int64_t *B = new int64_t[Col * Row];
    int64_t *B1 = new int64_t[Col * Row*1000];
    for (int i = 0; i < Col * Row; i++) {
        B[i] = rand() % 51;
    }
    for (int i = 0; i < Col * Row*1000; i++) {
        B1[i] = rand() % 51;
    }
    vector<int64_t*> Result;
    for(int i = 0; i < 5; i++) {
        int64_t *C = new int64_t[Row * Col/5];
        Result.push_back(C);
    }
    for(int i=0; i<100; i++){
        for(int i = 0; i < 2; i++) {
            for(int j=0;j<2;j++){
                InterChiplet::sendMessage(i, j, idX, idY, A1[i*2+j], Row/5 * Col * 1000 * sizeof(int64_t));
                InterChiplet::sendMessage(i, j, idX, idY, B1, Row * Col * 1000 * sizeof(int64_t));
            }
        }
        for(int i = 0; i < 2; i++) {
            for(int j=0;j<2;j++){
                InterChiplet::receiveMessage(idX, idY, i, j, Result[i*2+j], Row * (Col/5) * sizeof(int64_t));
            }
        }
    }
    for(int i = 0; i < 2; i++) {
        for(int j=0;j<2;j++){
            InterChiplet::sendMessage(i, j, idX, idY, A[i*2+j], Row/5 * Col * sizeof(int64_t));
            InterChiplet::sendMessage(i, j, idX, idY, B, Row * Col * sizeof(int64_t));
        }
    }
    InterChiplet::sendMessage(0, 3, idX, idY, A[4], Row/5 * Col * sizeof(int64_t));
    InterChiplet::sendMessage(0, 3, idX, idY, B, Row * Col * sizeof(int64_t));

    for(int i = 0; i < 2; i++) {
        for(int j=0;j<2;j++){
            InterChiplet::receiveMessage(idX, idY, i, j, Result[i*2+j], Row * (Col/5) * sizeof(int64_t));
        }
    }
    InterChiplet::receiveMessage(idX, idY, 0, 3, Result[4], Row * (Col/5) * sizeof(int64_t));
    // for(int i = 0; i < Row; i++) {
    //     for(int j = 0; j < Col; j++) {
    //         cout << Result[i/20][j+i%20*Col] << " ";
    //     }
    //     cout << endl;
    // }
    for(int i = 0; i < 5; i++) {
        delete[] A[i];
        delete[] Result[i];
    }
    delete[] B;
}
