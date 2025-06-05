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
    
    # 输出所有去重通信记录
    print("\n所有去重通信记录:")
    for i, record in enumerate(communication_records):
        print(f"{i+1}. 时间: {record['start_time']}, 源: {record['source']}, 目标: {record['destination']}, 延迟: {record['delay']}")
    
    # 构建节点通信时间线
    node_timeline = defaultdict(list)
    
    # 记录每个节点的接收和发送时间点
    for record in communication_records:
        start_time = record['start_time']
        source = record['source']
        destination = record['destination']
        
        # 记录源节点的发送时间
        node_timeline[source].append(('send', start_time, record))
        
        # 记录目标节点的接收时间
        node_timeline[destination].append(('receive', start_time, record))
    
    # 计算每个节点的计算时间
    computation_time = defaultdict(list)
    computation_details = defaultdict(list)
    
    # 对每个节点，按时间排序所有事件
    for node, events in node_timeline.items():
        # 按时间排序
        events.sort(key=lambda x: x[1])
        
        # 按顺序分析事件
        for i in range(len(events) - 1):
            current_event = events[i]
            next_event = events[i + 1]
            
            # 如果当前事件是接收，下一个事件是发送，则这是一个计算周期
            if current_event[0] == 'receive' and next_event[0] == 'send':
                recv_time = current_event[1]
                send_time = next_event[1]
                
                # 计算时间 = 发送start time - 接收start time
                comp_time = send_time - recv_time
                
                if comp_time > 0:
                    computation_time[node].append(comp_time)
                    
                    # 记录详细计算过程
                    computation_details[node].append({
                        'receive_time': recv_time,
                        'receive_record': current_event[2],
                        'send_time': send_time,
                        'send_record': next_event[2],
                        'computation_time': comp_time
                    })
    
    # 计算每个节点的总通信时间（作为目标节点的所有延迟之和）
    communication_time = defaultdict(int)
    for record in communication_records:
        communication_time[record['destination']] += record['delay']
    
    # 计算τ值
    tau_values = {}
    for node in set(communication_time.keys()):
        total_comp_time = sum(computation_time[node])
        total_comm_time = communication_time[node]
        
        if total_comm_time > 0:
            tau = total_comp_time / total_comm_time
            tau_values[node] = tau
        else:
            tau_values[node] = "N/A"
    
    # 输出各节点通信统计
    print("\n节点通信统计:")
    for node in sorted(set([rec['source'] for rec in communication_records] + [rec['destination'] for rec in communication_records])):
        as_source = sum(1 for rec in communication_records if rec['source'] == node)
        as_dest = sum(1 for rec in communication_records if rec['destination'] == node)
        print(f"节点 {node}: 作为源 {as_source} 次, 作为目标 {as_dest} 次")
    
    # 输出各节点计算与通信分析
    print("\n各节点计算与通信分析:")
    print("节点\t总计算时间\t总通信时间\tτ值\t瓶颈类型")
    print("-" * 70)
    
    total_comm_time_all = 0
    total_comp_time_all = 0
    for node in sorted(set(communication_time.keys())):
        total_comp_time = sum(computation_time[node])
        total_comm_time = communication_time[node]
        
        
        tau = tau_values[node]
        if tau != "N/A":
            bottleneck = "计算瓶颈" if tau > 1 else "通信瓶颈"
        else:
            bottleneck = "未知"
        
        if(node != cpu_node): 
            print(f"{node}\t{total_comp_time}\t{total_comm_time}\t{tau}\t{bottleneck}")
            total_comm_time_all += total_comm_time
            total_comp_time_all += total_comp_time
    
    # 计算整体τ值，先计算整体通信时间，再计算整体计算时间
    print(f"总计算时间: {total_comp_time_all}")
    print(f"总通信时间: {total_comm_time_all}")
    avg_tau = total_comp_time_all / total_comm_time_all
    print(f"平均τ值: {avg_tau}")
    
    # # 计算整体τ值，先计算整体通信时间，再计算整体计算时间
    # avg_tau = 0
    # for node in set(communication_time.keys()).union(set(computation_time.keys())):
    #     if node != cpu_node:
    #         avg_tau += tau_values[node]
    # avg_tau = avg_tau / len(set(communication_time.keys()).union(set(computation_time.keys())))
    # print(f"平均τ值: {avg_tau}")
    
    # 输出每个节点的计算时间详情
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
                recv = detail['receive_record']
                send = detail['send_record']
                print(f"  计算 {i+1}:")
                print(f"    接收: 时间 {detail['receive_time']}, 源 {recv['source']}, 目标 {recv['destination']}, 延迟 {recv['delay']}")
                print(f"    发送: 时间 {detail['send_time']}, 源 {send['source']}, 目标 {send['destination']}, 延迟 {send['delay']}")
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
    
    cpu_node = 35
    analyze_chiplet_communication_computation(log_file_path, cpu_node)