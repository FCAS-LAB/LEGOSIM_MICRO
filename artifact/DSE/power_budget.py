import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

df = pd.read_csv('optimization_results.csv')

df['T'] = df['T'].apply(lambda x: float(x.strip('[]')))

power_limits = sorted(df['Power_Limit'].unique())[:6]
df = df[df['Power_Limit'].isin(power_limits)]

max_t_per_power = df.groupby('Power_Limit')['T'].max().to_dict()

df['Normalized_T'] = df.apply(lambda row: row['T'] / max_t_per_power[row['Power_Limit']], axis=1)

normalized_data = {'Power_Limit': []}
for solution_type in ['Optimal', 'Non-Optimal 1', 'Non-Optimal 2']:
    normalized_data[solution_type] = []

for power_limit in power_limits:
    power_df = df[df['Power_Limit'] == power_limit]
    normalized_data['Power_Limit'].append(power_limit)
    
    for solution_type in ['Optimal', 'Non-Optimal 1', 'Non-Optimal 2']:
        solution_df = power_df[power_df['Solution_Type'] == solution_type]
        if not solution_df.empty:
            normalized_data[solution_type].append(solution_df['Normalized_T'].values[0])
        else:
            normalized_data[solution_type].append(np.nan)

normalized_df = pd.DataFrame(normalized_data)

plt.figure(figsize=(14, 6))
plt.rcParams.update({'font.size': 25})
bar_width = 0.25
index = np.arange(len(power_limits))

patterns = ['//', '\\\\', 'xx']
colors = ['#2F4F4F', '#87CEEB', '#DEB887']

bars = []
labels = ['Optimized results', 'Ref. config. 1', 'Ref. config. 2']
solution_types = ['Optimal', 'Non-Optimal 1', 'Non-Optimal 2']

for i, solution_type in enumerate(solution_types):
    bar = plt.bar(index + i*bar_width, normalized_df[solution_type], 
            bar_width, color=colors[i], edgecolor='black', 
            label=labels[i], hatch=patterns[i])
    bars.append(bar)

plt.xlabel('Power Limit (W)')
plt.ylabel('Norm. Exec. Time')
plt.xticks(index + bar_width, [f"{int(p)}" for p in power_limits],rotation=0)
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

plt.tight_layout()
plt.savefig('normalized_execution_time.png', dpi=300)

plt.show()
