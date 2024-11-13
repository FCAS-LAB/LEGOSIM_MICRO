#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include "apis_c.h"
#include <stack> 
#include "../../interchiplet/includes/pipe_comm.h"

InterChiplet::PipeComm global_pipe_comm;

#define numNodes 10

using namespace std;

// 子图的邻接表表示
vector<vector<int>> subgraph;


// 广度优先搜索遍历子图
void dfs(int start, int subgraph[numNodes][numNodes], bool visited[numNodes],std::stack<int> &q) {

    visited[start] = true; // 标记起始节点为已访问
    q.push(start); // 将起始节点压入栈中

    while (!q.empty()) { // 当栈不为空时
        int node = q.top(); // 获取栈顶的节点
        q.pop(); // 弹出栈顶的节点

        // 遍历当前节点的所有邻居
        for (int i = 0; i < numNodes; ++i) {
            if (subgraph[node][i] && !visited[i]) { // 如果邻居节点与当前节点相连且未被访问
                visited[i] = true; // 标记邻居节点为已访问
                q.push(i); // 将邻居节点压入栈中
            }
        }
    }
}

int main(int argc, char **argv)
{
    int idX = atoi(argv[1]);
    int idY = atoi(argv[2]);

    int(*subgraph)[numNodes] = (int(*)[numNodes])malloc(numNodes * numNodes * sizeof(int));
    bool *visited = (bool *)malloc(numNodes * sizeof(bool));
    for (int i = 0; i < numNodes; ++i)
    {
        visited[i] = false;
        for (int j = 0; j < numNodes; ++j)
        {
            subgraph[i][j] = 0;
        }
    }

    long long unsigned int time_end = 1;
    std::cout << "worker jieshouqian 1 1-------------------------------------" << std::endl;
    std::string fileName = InterChiplet::receiveSync(2, 2, idX, idY);
    global_pipe_comm.read_data(fileName.c_str(), subgraph, numNodes * numNodes * sizeof(int));
    time_end = InterChiplet::readSync(time_end, 2, 2, idX, idY, numNodes * numNodes * sizeof(int), 0);
    std::cout << "worker sendMessage 2 2 -------------------------------------" << std::endl;
    int startNode = 0; // 可以根据需要修改起始节点
   // std::queue<int> q;
    std::stack<int> stack; // 使用一个栈来存储待访问的节点
    dfs(startNode, subgraph, visited, stack);

    fileName = InterChiplet::sendSync(idX, idY, 2, 2);
         std::cout << "worker sendMessage 3 1-------------------------------------" << std::endl;
    global_pipe_comm.write_data(fileName.c_str(), visited, numNodes * sizeof(bool));
    InterChiplet::writeSync(time_end, idX, idY, 2, 2, numNodes * sizeof(bool), 0);
      std::cout << "worker wrieStlnc 3 2-------------------------------------" << std::endl;


    free(subgraph);
    free(visited);
    return 0;
}
