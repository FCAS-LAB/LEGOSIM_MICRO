#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "apis_c.h"

using namespace std;

// 自动生成随机图
vector<vector<int>> generateRandomGraph(int numNodes)
{
    vector<vector<int>> randomGraph(numNodes);
    srand(time(nullptr));
    for (int i = 0; i < numNodes; ++i)
    {
        for (int j = i + 1; j < numNodes; ++j)
        {
            if (rand() % 2)
            {
                randomGraph[i].push_back(j);
                randomGraph[j].push_back(i);
            }
        }
    }
    return randomGraph;
}

int main(int argc, char **argv)
{
    int idX = atoi(argv[1]);  //转换成整型
    int idY = atoi(argv[2]);

    int numNodes = 10;
    int(*graph)[numNodes] = (int(*)[numNodes])malloc(numNodes * numNodes * sizeof(int));
    bool *subgraph1 = (bool *)malloc(numNodes * sizeof(bool));
    bool *subgraph2 = (bool *)malloc(numNodes * sizeof(bool));
    bool *subgraph3 = (bool *)malloc(numNodes * sizeof(bool));
    bool *subgraph4 = (bool *)malloc(numNodes * sizeof(bool));
    bool *subgraph5 = (bool *)malloc(numNodes * sizeof(bool));
    std::vector<std::vector<int>> srcgraph = generateRandomGraph(numNodes);
    for (int i = 0; i < numNodes; ++i)
    {
        subgraph1[i] = false;
        subgraph2[i] = false;
        subgraph3[i] = false;
        subgraph4[i] = false;
        subgraph5[i] = false;
        for (int j = 0; j < numNodes; ++j)
        {
            graph[i][j] = srcgraph[i][j];
        }
    }
      std::cout << "master sendMessage 1 1-------------------------------------" << std::endl;
    InterChiplet::sendMessage(0, 0, idX, idY, graph, numNodes * numNodes * sizeof(int));
    InterChiplet::sendMessage(0, 1, idX, idY, graph, numNodes * numNodes * sizeof(int));
    InterChiplet::sendMessage(1, 0, idX, idY, graph, numNodes * numNodes * sizeof(int));
    InterChiplet::sendMessage(1, 1, idX, idY, graph, numNodes * numNodes * sizeof(int));
    InterChiplet::sendMessage(1, 2, idX, idY, graph, numNodes * numNodes * sizeof(int));

      std::cout << "master sendMessage 2 2-------------------------------------" << std::endl;
    InterChiplet::receiveMessage(idX, idY, 0, 0, subgraph1, numNodes * sizeof(bool));
    InterChiplet::receiveMessage(idX, idY, 0, 1, subgraph2, numNodes * sizeof(bool));
    InterChiplet::receiveMessage(idX, idY, 1, 0, subgraph3, numNodes * sizeof(bool));
    InterChiplet::receiveMessage(idX, idY, 1, 1, subgraph4, numNodes * sizeof(bool));
    InterChiplet::receiveMessage(idX, idY, 1, 2, subgraph5, numNodes * sizeof(bool));

    free(graph);
    free(subgraph1);
    free(subgraph2);
    free(subgraph3);
    free(subgraph4);
    free(subgraph5);
    return 0;
}
