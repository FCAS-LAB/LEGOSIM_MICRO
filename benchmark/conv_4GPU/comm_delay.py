import re
from collections import defaultdict

def analyze_chiplet_communication_computation(log_file_path, cpu_node=35):
    # 用于存储原始通信记录
    raw_records = []
    
    # 解析日志文件
    pattern = r'start time: (\d+) source address: (\d+) destination address: (\d+) delay: (\d+)'
    
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    start_time, source, destination, delay = match.groups()
                    start_time = int(start_time)
                    source = int(source)
                    destination = int(destination)
                    delay = int(delay)
                    
                    # 添加到原始记录
                    raw_records.append({
                        'start_time': start_time,
                        'source': source,
                        'destination': destination,
                        'delay': delay
                    })
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 找出每个start_time下最大延迟的记录
    max_delay_records = {}
    for record in raw_records:
        start_time = record['start_time']
        delay = record['delay']
        
        if start_time not in max_delay_records or delay > max_delay_records[start_time]['delay']:
            max_delay_records[start_time] = record
    
    # 转换为列表并按开始时间排序
    communication_records = list(max_delay_records.values())
    communication_records.sort(key=lambda x: x['start_time'])
    
    print(f"原始记录数: {len(raw_records)}")
    print(f"去重后记录数: {len(communication_records)}")
    print(f"CPU分发节点: {cpu_node} (将被排除在瓶颈分析之外)")
    
    # 输出所有去重通信记录
    print("\n所有去重通信记录:")
    for i, record in enumerate(communication_records):
        print(f"{i+1}. 时间: {record['start_time']}, 源: {record['source']}, 目标: {record['destination']}, 延迟: {record['delay']}")
    
    # 为每个节点构建通信时间线（包括作为源和目标的所有时间点）
    node_timeline = defaultdict(list)
    
    # 记录每个节点参与的所有通信（无论作为源还是目标）
    for i, record in enumerate(communication_records):
        source = record['source']
        destination = record['destination']
        start_time = record['start_time']
        delay = record['delay']
        
        # 记录源节点和目标节点的通信时间
        node_timeline[source].append((start_time, delay, i, 'source'))
        node_timeline[destination].append((start_time, delay, i, 'destination'))
    
    # 计算每个节点的计算时间
    computation_time = defaultdict(list)
    computation_details = defaultdict(list)
    
    # 对每个节点，按时间排序所有通信事件
    for node, events in node_timeline.items():
        # 按start_time排序
        events.sort(key=lambda x: x[0])
        
        # 计算相邻通信之间的时间差作为计算时间
        for i in range(len(events) - 1):
            current_time, current_delay, current_idx, current_role = events[i]
            next_time, next_delay, next_idx, next_role = events[i + 1]
            
            # 计算时间 = 下一次通信的start_time - (当前通信的start_time + 当前通信延迟的一半)
            adjusted_current_time = current_time + (current_delay // 2)
            comp_time = next_time - adjusted_current_time
            
            # 如果计算时间小于0，则设为0
            comp_time = max(0, comp_time)
            
            # 只记录计算时间
            computation_time[node].append(comp_time)
            
            # 记录详细计算过程
            current_record = communication_records[current_idx]
            next_record = communication_records[next_idx]
            
            computation_details[node].append({
                'start_event': {
                    'time': current_time,
                    'adjusted_time': adjusted_current_time,
                    'record': current_record,
                    'role': current_role
                },
                'end_event': {
                    'time': next_time,
                    'record': next_record,
                    'role': next_role
                },
                'computation_time': comp_time
            })
    
    # 计算每个节点的总通信时间（作为目标节点的所有延迟之和）
    communication_time = defaultdict(int)
    for record in communication_records:
        communication_time[record['destination']] += record['delay']
    
    # 计算τ值（排除CPU节点）
    tau_values = {}
    
    for node in set(communication_time.keys()).union(set(computation_time.keys())):
        if node == cpu_node:
            continue  # 跳过CPU节点
            
        total_comp_time = sum(computation_time[node])
        total_comm_time = communication_time.get(node, 0)
        
        if total_comm_time > 0:
            tau = total_comp_time / total_comm_time
            tau_values[node] = tau
        else:
            if total_comp_time > 0:
                tau_values[node] = "∞ (无通信延迟)"
            else:
                tau_values[node] = "N/A"
    
    # 输出各节点通信统计
    print("\n节点通信统计:")
    for node in sorted(set([rec['source'] for rec in communication_records] + [rec['destination'] for rec in communication_records])):
        as_source = sum(1 for rec in communication_records if rec['source'] == node)
        as_dest = sum(1 for rec in communication_records if rec['destination'] == node)
        print(f"节点 {node}: 作为源 {as_source} 次, 作为目标 {as_dest} 次")
    
    # 输出各节点计算与通信分析（排除CPU节点的瓶颈分析）
    print("\n各节点计算与通信分析:")
    print("节点\t总计算时间\t总通信时间\tτ值\t瓶颈类型")
    print("-" * 70)
    
    for node in sorted(set(communication_time.keys()).union(set(computation_time.keys()))):
        total_comp_time = sum(computation_time[node])
        total_comm_time = communication_time.get(node, 0)
        
        tau = tau_values.get(node, "CPU节点")
        
        if node == cpu_node:
            bottleneck = "不分析"
        elif tau != "N/A" and tau != "∞ (无通信延迟)" and tau != "CPU节点":
            # 简化瓶颈判断: τ > 1 为计算瓶颈, τ < 1 为通信瓶颈
            bottleneck = "计算瓶颈" if float(tau) > 1 else "通信瓶颈"
        elif tau == "∞ (无通信延迟)":
            bottleneck = "计算瓶颈"
        else:
            bottleneck = "未知"
        
        print(f"{node}\t{total_comp_time}\t{total_comm_time}\t{tau}\t{bottleneck}")
    
    # 输出每个节点的计算时间详情（包括CPU节点的计算时间，但不分析其瓶颈）
    print("\n各节点计算时间详情:")
    for node in sorted(computation_time.keys()):
        times = computation_time[node]
        if times:
            print(f"节点 {node} 的计算时间: {times}")
            print(f"节点 {node} 的计算次数: {len(times)}")
            print(f"节点 {node} 的平均计算时间: {sum(times)/len(times)}")
            
            # 输出详细计算过程
            print(f"节点 {node} 的详细计算过程:")
            for i, detail in enumerate(computation_details[node]):
                start_event = detail['start_event']
                end_event = detail['end_event']
                start_record = start_event['record']
                end_record = end_event['record']
                
                print(f"  计算 {i+1}:")
                print(f"    开始事件: 时间 {start_event['time']}, 调整后时间 {start_event['adjusted_time']}, 角色 {start_event['role']}")
                print(f"    开始通信: 源 {start_record['source']}, 目标 {start_record['destination']}, 延迟 {start_record['delay']}")
                print(f"    结束事件: 时间 {end_event['time']}, 角色 {end_event['role']}")
                print(f"    结束通信: 源 {end_record['source']}, 目标 {end_record['destination']}, 延迟 {end_record['delay']}")
                print(f"    计算公式: {end_event['time']} - {start_event['adjusted_time']} = {detail['computation_time']}")
                print(f"    计算时间: {detail['computation_time']}")
            
            print("-" * 30)
    
    return {
        'communication_records': communication_records,
        'computation_time': dict(computation_time),
        'communication_time': dict(communication_time),
        'tau_values': tau_values
    }

if __name__ == "__main__":
    # 替换为你的日志文件路径
    log_file_path = "./proc_r1_p2_t0/popnet_0.log"
    
    # 指定CPU分发节点的地址（默认为35）
    cpu_node = 35
    
    analyze_chiplet_communication_computation(log_file_path, cpu_node)