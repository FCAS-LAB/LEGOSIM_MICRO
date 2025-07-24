#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include "apis_c.h"
#include "graph.h"

void random_init(int64_t * data, int size){
    int i;
    for (i = 0; i < size; i++){
        //data[i] = rand();
        data[i] = 128;
    }
}

int main(int argc, char **argv){
    int idX = atoi(argv[1]);
    int idY = atoi(argv[2]);
    int no_of_nodes = 6;
    int edge_list_size = 8;

    int h_graph_nodes_start[6] = {
        0, 2, 3, 4, 5, 6
    };
    int h_graph_nodes_no_of_edges[6] = {
        2, 1, 1, 1, 1, 1
    };
    // Node h_graph_nodes[6] = {
    //     {0, 2}, {2, 1}, {3, 1}, {4, 1}, {5, 1}, {6, 1}
    // };

    int h_graph_edges[8] = {
        1, 2, 3, 4, 5, 0, 0, 0
    };

    int64_t input = 1;
    InterChiplet::sendMessage(1, 2, idX, idY, &input, sizeof(int64_t));
    float* rec_a = new float[10*10*10*10];
    InterChiplet::receiveMessage(idX, idY, 1, 2, rec_a, sizeof(float)*10*10*10*10);
    float* rec_b = new float[10*10*10*5];
    InterChiplet::receiveMessage(idX, idY, 1, 2, rec_b, sizeof(float)*10*10*10*5);

    int size = 12*12*8;
    int64_t *test = new int64_t[size];
    int64_t *test_ans = new int64_t[size];
    random_init(test, size);
    InterChiplet::sendMessage(1, 1, idX, idY, test, size*sizeof(int64_t));
    InterChiplet::receiveMessage(idX, idY, 1, 1, test_ans, size*sizeof(int64_t));
    delete[] test;
    delete[] test_ans;
    std::cout<<"-------------------------------------mnsim and dsa over-------------------------------------"<<std::endl;

    bool h_graph_mask[6] = {true, false, false, false, false, false};
    bool h_graph_visited[6] = {false, false, false, false, false, false};
    int h_cost[6] = {-1, -1, -1, -1, -1, -1};
    int h_cost_2[6] = {-1, -1, -1, -1, -1, -1};
    int h_cost_3[6] = {-1, -1, -1, -1, -1, -1};
    int h_cost_4[6] = {-1, -1, -1, -1, -1, -1};
    h_cost[0] = 0;
    h_cost_2[0] = 0;
    h_cost_3[0] = 0;
    h_cost_4[0] = 0;

    InterChiplet::sendMessage(0, 0, idX, idY, h_graph_nodes_start, no_of_nodes * sizeof(int));
    InterChiplet::sendMessage(0, 0, idX, idY, h_graph_nodes_no_of_edges, no_of_nodes * sizeof(int));
    InterChiplet::sendMessage(0, 0, idX, idY, h_graph_edges, edge_list_size * sizeof(int));
    InterChiplet::sendMessage(0, 0, idX, idY, h_graph_mask, no_of_nodes * sizeof(bool));
    InterChiplet::sendMessage(0, 0, idX, idY, h_graph_visited, no_of_nodes * sizeof(bool));
    InterChiplet::sendMessage(0, 0, idX, idY, h_cost, no_of_nodes * sizeof(int));

    InterChiplet::sendMessage(0, 1, idX, idY, h_graph_nodes_start, no_of_nodes * sizeof(int));
    InterChiplet::sendMessage(0, 1, idX, idY, h_graph_nodes_no_of_edges, no_of_nodes * sizeof(int));
    InterChiplet::sendMessage(0, 1, idX, idY, h_graph_edges, edge_list_size * sizeof(int));
    InterChiplet::sendMessage(0, 1, idX, idY, h_graph_mask, no_of_nodes * sizeof(bool));
    InterChiplet::sendMessage(0, 1, idX, idY, h_graph_visited, no_of_nodes * sizeof(bool));
    InterChiplet::sendMessage(0, 1, idX, idY, h_cost, no_of_nodes * sizeof(int));

    InterChiplet::sendMessage(0, 2, idX, idY, h_graph_nodes_start, no_of_nodes * sizeof(int));
    InterChiplet::sendMessage(0, 2, idX, idY, h_graph_nodes_no_of_edges, no_of_nodes * sizeof(int));
    InterChiplet::sendMessage(0, 2, idX, idY, h_graph_edges, edge_list_size * sizeof(int));
    InterChiplet::sendMessage(0, 2, idX, idY, h_graph_mask, no_of_nodes * sizeof(bool));
    InterChiplet::sendMessage(0, 2, idX, idY, h_graph_visited, no_of_nodes * sizeof(bool));
    InterChiplet::sendMessage(0, 2, idX, idY, h_cost, no_of_nodes * sizeof(int));

    InterChiplet::sendMessage(1, 0, idX, idY, h_graph_nodes_start, no_of_nodes * sizeof(int));
    InterChiplet::sendMessage(1, 0, idX, idY, h_graph_nodes_no_of_edges, no_of_nodes * sizeof(int));
    InterChiplet::sendMessage(1, 0, idX, idY, h_graph_edges, edge_list_size * sizeof(int));
    InterChiplet::sendMessage(1, 0, idX, idY, h_graph_mask, no_of_nodes * sizeof(bool));
    InterChiplet::sendMessage(1, 0, idX, idY, h_graph_visited, no_of_nodes * sizeof(bool));
    InterChiplet::sendMessage(1, 0, idX, idY, h_cost, no_of_nodes * sizeof(int));

    InterChiplet::receiveMessage(idX, idY, 0, 0, h_cost, no_of_nodes * sizeof(int));
    InterChiplet::receiveMessage(idX, idY, 0, 1, h_cost_2, no_of_nodes * sizeof(int));
    InterChiplet::receiveMessage(idX, idY, 0, 2, h_cost_3, no_of_nodes * sizeof(int));
    InterChiplet::receiveMessage(idX, idY, 1, 0, h_cost_4, no_of_nodes * sizeof(int));

    for (int i = 0; i < no_of_nodes; i++) {
        std::cout << "Node " << i << " cost: " << h_cost[i] << std::endl;
    }
}