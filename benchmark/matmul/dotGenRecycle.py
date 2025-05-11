####  依据 bench.txt文件数据传输记录生成GraphViz（dot）任务图==> *.gv, *.gv.jpg
"""
****** 依据 bench.txt文件数据传输记录生成GraphViz（dot）任务图==> *.gv, *.gv.jpg
Bench.txt format :
from popnet_chiplet\sim_protoengine.h
// changed at 2024-6-4
struct ProtoPacket {
    time_type srcTime, dstTime;
    add_type sourceAddress, destinationAddress;
    long packetSize;
    long protoDesc;
    // 用于数据包的唯一标识
    typedef size_t TId;
    TId id;

    ProtoPacket() {
        size_t addSize = configuration::ap().cube_number();
        sourceAddress.reserve(addSize);
        destinationAddress.reserve(addSize);
    }
};
"""

"""
# 读取BenchMark的bench流量数据传输日志文件
"""
from graphviz import Digraph
import numpy as np
import ReadBench as ReadBench

# Read benchmark data
bench_file = "bench_mesh.txt"
benchList = ReadBench.read_formatted_txt(bench_file)
benchArray = np.array(benchList)
selInd = [2, 3, 4, 5, 6, 0, 1]
useBench = benchArray[:, selInd]
print(useBench)

# Create the main Digraph with larger font sizes
grap_g = Digraph("G", format="jpg")
grap_g.attr(fontsize="30")  # Set default graph font size to 30
grap_g.node_attr.update(fontsize="28")  # Set default node font size to 28
grap_g.edge_attr.update(fontsize="25")  # Set default edge font size to 25

# Create subgraph with larger font sizes and styles
sub_g0 = Digraph(
    comment="bench.txt",
    graph_attr={"style": 'filled', "color": 'lightgrey'},
    node_attr={"style": "filled", "color": "red", "fontsize": "28"}
)

# Construct graph nodes
nodepush = []
for row in range(useBench.shape[0]):
    nodeSrc = f"{useBench[row, 0]}_{useBench[row, 1]}"
    nodeDst = f"{useBench[row, 2]}_{useBench[row, 3]}"
    if nodeSrc not in nodepush:
        nodepush.append(nodeSrc)
        sub_g0.node(nodeSrc, f"{useBench[row, 0]}:{useBench[row, 1]}")
    if nodeDst not in nodepush:
        nodepush.append(nodeDst)
        sub_g0.node(nodeDst, f"{useBench[row, 2]}:{useBench[row, 3]}")

# Draw edges based on each row of data
for row in range(useBench.shape[0]):
    nodeSrc = f"{useBench[row, 0]}_{useBench[row, 1]}"
    nodeDst = f"{useBench[row, 2]}_{useBench[row, 3]}"
    edge = f"{useBench[row, 4]}"
    sub_g0.edge(nodeSrc, nodeDst, label=edge)

# Add start and end nodes with custom font sizes
grap_g.node("start", label="start", shape="Mdiamond", fillcolor="skyblue", fontsize="28")
grap_g.node("end", label="end", shape="Mdiamond", fillcolor="black", fontsize="28")

grap_g.subgraph(sub_g0)

# Connect start and end nodes
entryNode = f"{useBench[0, 0]}_{useBench[0, 1]}"
endNode = f"{useBench[useBench.shape[0]-1, 2]}_{useBench[useBench.shape[0]-1, 3]}"
grap_g.edge("start", entryNode)
grap_g.edge(endNode, "end", constraint='false')

# Render the graph
grap_g.render('./out/task.gv')



