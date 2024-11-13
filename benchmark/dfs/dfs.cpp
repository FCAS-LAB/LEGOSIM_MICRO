#include <iostream>
#include <vector>
#include <stack>
#include <thread>
#include <mutex>
#include <list>
#include <unordered_map>
#include <condition_variable>
#include <atomic>

class Graph {
public:
    std::unordered_map<int, std::list<int>> adj;

    void addEdge(int v, int w) {
        adj[v].push_back(w);  // 添加一个从v到w的边
    }
};

void dfs(Graph const& graph, int start_vertex, std::vector<bool>& visited) {
    std::stack<int> stack;
    stack.push(start_vertex);

    while (!stack.empty()) {
        int vertex = stack.top();
        stack.pop();

        if (!visited[vertex]) {
            visited[vertex] = true;
            std::cout << "Visited " << vertex << std::endl;

            for (auto it = graph.adj[vertex].rbegin(); it != graph.adj[vertex].rend(); ++it) {
                if (!visited[*it]) {
                    stack.push(*it);
                }
            }
        }
    }
}

void parallel_dfs(Graph const& graph, const std::vector<int>& start_points) {
    std::vector<bool> visited(graph.adj.size(), false);
    std::vector<std::thread> threads;
    std::mutex io_mutex;

    for (int start_vertex : start_points) {
        threads.push_back(std::thread([&graph, &visited, start_vertex, &io_mutex]() {
            dfs(graph, start_vertex, visited);
        }));
    }

    for (auto& t : threads) {
        t.join();
    }
}

int main() {
    Graph g;
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 2);
    g.addEdge(2, 0);
    g.addEdge(2, 3);
    g.addEdge(3, 3);

    // 可以从多个点启动DFS以展示并行性，例如从每个孤立部分的某个点
    std::vector<int> start_points = {0, 2};  // 假设图中有从0和2可以独立进行DFS的部分
    parallel_dfs(g, start_points);

    return 0;
}
