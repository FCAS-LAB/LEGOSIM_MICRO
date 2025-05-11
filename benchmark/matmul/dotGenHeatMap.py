####  依据 pacCountInfo.txt文件数据传输记录生成GraphViz（dot）任务图==> *.gv, *.gv.jpg
"""
****** 依据 pacCountInfo.txt文件数据传输记录生成GraphViz（dot）任务图==> *.gv, *.gv.jpg
pacCountInfo.txt format :
from popnet_chiplet\sim_protoengine.h
// changed at 2024-6-4
struct ProtoPacket {
    time_type srcTime, dstTime;
    add_type sourceAddress, destinationAddress;
    long packetTimes;
};
"""
"""画热图函数"""
import matplotlib.pyplot as plt
def Heat(data, title):
    plt.imshow(data, cmap='hot',interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.savefig('./heatmap/'+title+'.jpg')
    # plt.imsave('./heatmap/'+title+'1.png')
    plt.show()
    plt.close()
"""
# 读取BenchMark的bench流量数据传输日志文件
"""
import ReadBench as ReadBench
bench_file = "pacCountInfo.txt"
# benchList = ReadBench.read_formatted_txt()
benchList = ReadBench.read_formatted_txt(bench_file)

import numpy as numpy
benchArray = numpy.array(benchList)
selInd = [2, 3, 4, 5, 6, 7, 0, 1]
useBench = benchArray[:, selInd] # Python中的列索引从0开始
print(useBench)
#对数据进行筛选、清理以及整和

"""统计芯片芯粒阵列"""
maxIndx = 0
maxIndy = 0
for row in range(useBench.shape[0]):
    inds_x = int(useBench[row, 0])
    inds_y = int(useBench[row, 1])
    indd_x = int(useBench[row, 2])
    indd_y = int(useBench[row, 3])
    maxIndx = max(maxIndx, inds_x, indd_x)
    maxIndy = max(maxIndy, inds_y, indd_y)

"""
  Graphy 生成图
draw a Digraph into *.gv and *.jpg
"""
#循环获取芯粒 0:0
rows = maxIndx+1
columns = maxIndy+1
nodesend = [[0 for _ in range(columns)] for _ in range(rows)]
noderecv = [[0 for _ in range(columns)] for _ in range(rows)]
for row in range(useBench.shape[0]):
    inds_x = int(useBench[row, 0])
    inds_y = int(useBench[row, 1])
    indd_x = int(useBench[row, 2])
    indd_y = int(useBench[row, 3])

    timeCount = int(useBench[row, 4])
    pacSize = int(useBench[row, 5])

    nodesend[inds_x][inds_y] = nodesend[inds_x][inds_y] + timeCount
    noderecv[indd_x][indd_y] = noderecv[indd_x][indd_y] + timeCount

print('Send :')
for row in nodesend:
    print(row)
print('Recv :')
for row in noderecv:
    print(row)

Heat(nodesend, 'HyperHeterogeneousChip_Send')  # 超异构芯粒Send--发送
Heat(noderecv, 'HyperHeterogeneousChip_Recv')  # 超异构芯粒Recv--接收

node_total = [[nodesend[i][j] + noderecv[i][j] for j in range(columns)] for i in range(rows)]

# 打印总和矩阵（可选）
print('Total :')
for row in node_total:
    print(row)

# 绘制总和图
Heat(node_total, 'HyperHeterogeneousChip_Total')  # 总和图