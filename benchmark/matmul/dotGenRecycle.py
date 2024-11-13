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
import ReadBench as ReadBench
# bench_file = "bench_Matmul3.txt"
# bench_file = "bench_origin.txt"
bench_file = "bench(4).txt"
# bench_file = "benchLittle.txt"
# benchList = ReadBench.read_formatted_txt()
benchList = ReadBench.read_formatted_txt(bench_file)

import numpy as numpy
benchArray = numpy.array( benchList )
selInd = [2, 3, 4, 5, 6, 0, 1]
useBench = benchArray[:, selInd] # Python中的列索引从0开始
print(useBench)
"""
  Graphy 生成图
draw a Digraph into *.gv and *.jpg
"""
from graphviz import Digraph
grap_g = Digraph("G", format="jpg", graph_attr={"fontsize": "80"})  # , rankdir="BT"
# grap_g.attr(rankdir='BT')  # LR
# grap_g.attr(rankdir='LR')
sub_g0 = Digraph(comment="bench.txt",  graph_attr={"style":'filled',"color":'lightgrey',"fontsize": "80"},node_attr={"style":"filled","color":"red"})
# sub_g0.attr(rankdir='LR')  # LR

# first 构造图节点
# 初始节点 0:0

#循环获取芯粒 0:0
nodepush = []
for row in range(useBench.shape[0]):
    #for j in range(useBench.shape[1]):
    nodeSrc = f"{useBench[row, 0]}_{useBench[row, 1]}"
    nodeDst = f"{useBench[row, 2]}_{useBench[row, 3]}"
    if nodeSrc not in nodepush :
        nodepush.append(nodeSrc)
        # sub_g0.node(f"{row}_{j}",f"{row}:{j}")
        sub_g0.node(nodeSrc, f"{useBench[row, 0]}:{useBench[row, 1]}")

    if nodeDst not in nodepush :
        nodepush.append(nodeDst)
        # sub_g0.node(f"{row}_{j}",f"{row}:{j}")
        sub_g0.node(nodeDst, f"{useBench[row, 2]}:{useBench[row, 3]}")

# 依据每行数据 画边------ 循环图
lineDis = True
for row in range(useBench.shape[0]):
    nodeSrc = f"{useBench[row, 0]}_{useBench[row, 1]}"
    nodeDst = f"{useBench[row, 2]}_{useBench[row, 3]}"
    edge = useBench[row, 4]
    if lineDis:
        edge = f"{edge}({row+1})"
    else:
        # edge = f"{edge}-{useBench[row, 5]}-{useBench[row, 6]}"
        edge = f"{edge}\n({useBench[row, 5]})"  # 发出时序
    sub_g0.edge(nodeSrc,nodeDst, edge)
#first
grap_g.node(
"start", label="start",shape="Mdiamond", fillcolor="skyblue", fontsize="30")
grap_g.node(
"end", label="end", shape="Mdiamond", fillcolor="black", fontsize="30")

grap_g.subgraph(sub_g0)
entryNode = f"{useBench[0, 0]}_{useBench[0, 1]}"
endNode = f"{useBench[useBench.shape[0]-1, 2]}_{useBench[useBench.shape[0]-1, 3]}"
grap_g.edge("start",entryNode)
grap_g.edge(endNode, "end",  constraint='false')  # 确保边不会影响节点的实际位置

grap_g.render('./out/task.gv')  # , view=True
# grap_g.view()
# g = Digraph('测试图片', format='jpg')


