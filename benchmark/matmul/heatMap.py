import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 文件路径
log_file_path = "./proc_r1_p2_t0/popnet.log"

# 正则表达式模式，用于提取 router 的坐标
pattern = r"From Router (\d+) (\d+) to Router (\d+) (\d+)"

# 初始化一个字典，用于记录每个 router 的流量
traffic_dict = {}

# 从 log 文件中读取数据并解析
with open(log_file_path, "r") as file:
    for line in file:
        match = re.search(pattern, line)
        if match:
            src_x, src_y, dst_x, dst_y = map(int, match.groups())
            src = (src_x, src_y)
            dst = (dst_x, dst_y)
            
            # 更新源 router 的流量计数
            if src not in traffic_dict:
                traffic_dict[src] = {}
            if dst not in traffic_dict[src]:
                traffic_dict[src][dst] = 0
            traffic_dict[src][dst] += 1

# 找到最大坐标，用于定义矩阵的大小
max_x = max(max(src[0], dst[0]) for src in traffic_dict for dst in traffic_dict[src])
max_y = max(max(src[1], dst[1]) for src in traffic_dict for dst in traffic_dict[src])

# 初始化通信流量矩阵
traffic_matrix = np.zeros((max_x + 1, max_y + 1))

# 将流量数据填入矩阵
for src, dst_dict in traffic_dict.items():
    src_x, src_y = src
    for dst, count in dst_dict.items():
        traffic_matrix[src_x, src_y] += count

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(traffic_matrix, annot=True, cmap="YlOrRd", cbar_kws={'label': 'Traffic Volume'})
plt.title("Router Traffic Heatmap")
plt.xlabel("Router X Coordinate")
plt.ylabel("Router Y Coordinate")
# 反转 y 轴，使其从下往上显示
plt.gca().invert_yaxis()
plt.savefig("router_traffic_heatmap.png", dpi=300, bbox_inches='tight')