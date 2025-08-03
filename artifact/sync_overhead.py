import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

plt.rcParams.update({'font.size': 25})
plt.rcParams.update({
    "text.usetex": False,
})

schemes = ['PC','TQ-2', 'TQ-4', 'TQ-8', 'TQ-16', 'TQ-32', 'TQ-100', 'TQ-1000', 'On-demand']
sync_counts_raw = [57964800,28982400, 14491200, 7245600, 3618300, 1809100, 578900, 57800, 196]  

max_sync_count = max(sync_counts_raw)
sync_counts = [count/max_sync_count for count in sync_counts_raw]

intervals = [
    [0, 200, 2],
    [200, 400, 44],
    [400, 600, 15],
    [600, 800, 6],
    [800, 1000, 10],
    [1000, 1200, 1],
    [1200, 1400, 0],
    [1400, 1600, 0],
    [1600, 1800, 1],
    [1800, 2000, 0],
    [2000, float('inf'), 18]
]

total_samples = sum(interval[2] for interval in intervals)

tq_values = [1,2, 4, 8, 16, 32, 100, 1000]
weighted_delays = []

for tq in tq_values:
    weighted_delay = 0
    total_sync_count = 0
    for interval in intervals:
        start, end, count = interval
        if end >= 2000:
            median = int(2000 + np.random.uniform(-100, 500))
        else:
            median = int((start + end) / 2 + np.random.uniform(-10, 50))
        if median > tq:
            delay = median % tq
        else:
            delay = tq - median
        weighted_delay += delay * count
        total_sync_count += count * median
    weighted_delay = (1 - total_sync_count / (total_sync_count + weighted_delay))*100
    weighted_delays.append(weighted_delay)
    print(weighted_delay)

weighted_delays.append(0)

plt.figure(figsize=(14, 8))

ax1 = plt.gca()
ax2 = ax1.twinx()

x = np.arange(len(schemes))
width = 0.35
patterns = ['//', '\\\\']
colors = ['#2F4F4F', '#DEB887']
bars1 = ax1.bar(x - width/2, sync_counts, width, label='Norm. Sync Count', color=colors[0], hatch=patterns[0])

bars2 = ax2.bar(x + width/2, weighted_delays, width, label='$\epsilon_{sync}$ (%)', color=colors[1], hatch=patterns[1])

ax1.set_ylabel('Norm. Sync Time', fontsize=25)
ax2.set_ylabel('$\epsilon_{sync}$ (%)', fontsize=25)
ax1.set_xticks(x)
ax1.set_xticklabels(schemes, fontsize=25, rotation=15, ha='right')

ax1.set_ylim(0, 1.1)
ax2.set_ylim(0, max(weighted_delays) * 1.4)

ax1.grid(axis='y', linestyle='--', alpha=0.7)

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height < 0.0005 and height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'0',
                    ha='center', va='bottom', fontsize=15)
        elif height == 0:
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    '0',
                    ha='center', va='bottom', fontsize=15)

add_labels(bars1)
add_labels(bars2)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, 
           loc='upper right', fontsize=25, ncol=2)

plt.tight_layout()

plt.savefig('sync_overhead.png', dpi=300, bbox_inches='tight')
