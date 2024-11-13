#include <iostream>
#include <thread>
#include <mutex>
#include <climits>
#include "apis_c.h"

using namespace std;

// Graph representation as an adjacency list using an array
int graph[100][100];
int visited[100];
int num_threads;
int shortest_path = INT_MAX;
mutex mtx;

// Depth First Search (DFS) function to find the shortest path
void dfs(int node, int destination, int current_path_length) {
    // If the destination node is reached
    if (node == destination) {
        // Lock the mutex to safely update the shortest_path
        lock_guard<mutex> lock(mtx);
        if (current_path_length < shortest_path) {
            shortest_path = current_path_length;
        }
        return;
    }

    // Mark the current node as visited
    visited[node] = true;
    // Iterate over all neighbors of the current node
    for (int i = 0; i < 100; ++i) {
        if (graph[node][i] == 1 && !visited[i]) {
            dfs(i, destination, current_path_length + 1);
        }
    }
    // Backtrack: unmark the current node as visited
    visited[node] = false;
}

// Function to initiate parallel DFS from the start node
void parallel_dfs(int start_node, int destination) {
    thread threads[100];
    // Mark the start node as visited
    visited[start_node] = true;

    // Create a thread for each neighbor of the start node
    int thread_count = 0;
    for (int i = 0; i < 100; ++i) {
        if (graph[start_node][i] == 1 && !visited[i]) {
            threads[thread_count++] = thread(dfs, i, destination, 1);
        }
    }

    // Wait for all threads to finish
    for (int i = 0; i < thread_count; ++i) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }
}

int main() {
    int nodes = 100;

    // Initialize the graph and visited array
    for (int i = 0; i < nodes; ++i) {
        visited[i] = false;
        for (int j = 0; j < nodes; ++j) {
            graph[i][j] = 0;
        }
    }

    // Predefined edges for the graph
    int edges[][2] = {
        {0, 1}, {0, 2}, {1, 3}, {1, 4}, {2, 5}, {2, 6},
        {3, 7}, {4, 8}, {5, 9}, {6, 10}, {7, 11}, {8, 12},
        {9, 13}, {10, 14}, {11, 15}, {12, 16}, {13, 17}, {14, 18},
        {15, 19}, {16, 20}, {17, 21}, {18, 22}, {19, 23}, {20, 24},
        {21, 25}, {22, 26}, {23, 27}, {24, 28}, {25, 29}, {26, 30},
        {27, 31}, {28, 32}, {29, 33}, {30, 34}, {31, 35}, {32, 36},
        {33, 37}, {34, 38}, {35, 39}, {36, 40}, {37, 41}, {38, 42},
        {39, 43}, {40, 44}, {41, 45}, {42, 46}, {43, 47}, {44, 48},
        {45, 49}, {46, 50}, {47, 51}, {48, 52}, {49, 53}, {50, 54},
        {51, 55}, {52, 56}, {53, 57}, {54, 58}, {55, 59}, {56, 60},
        {57, 61}, {58, 62}, {59, 63}, {60, 64}, {61, 65}, {62, 66},
        {63, 67}, {64, 68}, {65, 69}, {66, 70}, {67, 71}, {68, 72},
        {69, 73}, {70, 74}, {71, 75}, {72, 76}, {73, 77}, {74, 78},
        {75, 79}, {76, 80}, {77, 81}, {78, 82}, {79, 83}, {80, 84},
        {81, 85}, {82, 86}, {83, 87}, {84, 88}, {85, 89}, {86, 90},
        {87, 91}, {88, 92}, {89, 93}, {90, 94}, {91, 95}, {92, 96},
        {93, 97}, {94, 98}, {95, 99}
    };

    // Add the edges to the graph (assuming an undirected graph)
    int num_edges = sizeof(edges) / sizeof(edges[0]);
    for (int i = 0; i < num_edges; ++i) {
        int u = edges[i][0];
        int v = edges[i][1];
        graph[u][v] = 1;
        graph[v][u] = 1;
    }

    int start_node = 0;
    int destination = 99;

    // Start the parallel DFS from the start node to the destination node
    parallel_dfs(start_node, destination);

    // Output the result
    if (shortest_path == INT_MAX) {
        cout << "No path found." << endl;
    } else {
        cout << "Shortest path length: " << shortest_path << endl;
    }

    return 0;
}