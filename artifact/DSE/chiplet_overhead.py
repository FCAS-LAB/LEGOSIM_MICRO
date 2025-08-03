import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

data = {
    'Chiplet': ['(0,0)', '(1,1)', '(2,0)', '(3,1)', 
                '(4,0)', '(5,0)', '(0,1)', '(1,0)', 
                '(2,1)', '(3,0)'],
    'Computation': [0.38, 0.26, 0.6, 0.55, 0.4, 0.23, 0.21, 0.18, 0.17, 0.15],
    'NoI': [0.76, 0.6, 0.65, 0.58, 0.57, 0.55, 0.69, 0.74, 0.54, 0.77],
    'Buffer_access': [0.63, 0.54, 0.6, 0.6, 0.5, 0.43, 0.59, 0.68, 0.57, 0.64]
}

print(data['Computation'])

df = pd.DataFrame(data)

categories = ['Computation', 'NoI', 'Buffer_access']
all_values = []
for category in categories:
    all_values.extend(df[category].tolist())
max_value = max(all_values)

for category in categories:
    df[f'Normalized_{category}'] = df[category] / max_value

plt.figure(figsize=(12, 6))
plt.rcParams.update({'font.size': 20})
bar_width = 0.25
index = np.arange(len(df['Chiplet']))

patterns = ['//', '\\\\', 'xx']
colors = ['#2F4F4F', '#87CEEB', '#DEB887']

bars = []
labels = ['Computation', 'NoI', 'Buffer access']
normalized_categories = ['Normalized_Computation', 'Normalized_NoI', 'Normalized_Buffer_access']

for i, category in enumerate(normalized_categories):
    bar = plt.bar(index + i*bar_width, df[category], 
            bar_width, color=colors[i], edgecolor='black', 
            label=labels[i], hatch=patterns[i])
    bars.append(bar)

plt.xlabel('Chiplet #')
plt.ylabel('Norm. Time')
plt.xticks(index + bar_width, [f"{df['Chiplet'][i]}" for i in range(len(df['Chiplet']))], rotation=45)
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=3)

plt.tight_layout()
plt.savefig('chiplet_performance_normalized.png', dpi=300)

plt.show()
