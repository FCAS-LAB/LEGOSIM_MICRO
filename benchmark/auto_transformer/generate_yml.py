import csv
import argparse

def generate_yaml_file_from_csv(csv_path="device_map.csv", yml_path="mlp_config.yml", width=2, height=2, GV_path="mesh_{width}_{height}.gv", popnet_clock_rate=3):
    try:
        x_cpu = 0
        y_cpu = 0
        with open(csv_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            print(f"CSV文件: {csv_path}")
            # 打印CSV的所有行用于调试
            rows = list(reader)
            for row in rows:
                print(f"行: {row}")
            
            # 重新构建映射
            device_map = {}
            clock_rate_map = {}
            for row in rows:
                try:
                    node_id = int(row["node"])
                    device_map[node_id] = row["device"].strip().upper()
                    clock_rate_map[node_id] = int(row["clock rate"])
                except Exception as e:
                    print(f"处理行时出错: {row}, 错误: {e}")
            
            # 调试输出
            print(f"设备映射: {device_map}")
            print(f"时钟频率映射: {clock_rate_map}")
            for node_id, device in device_map.items():
                if device == "CPU":
                    x_cpu = node_id // width
                    y_cpu = node_id % width
        
        
        with open(yml_path, 'w') as file:
            file.write("# 自动生成的配置文件\n")
            file.write("phase1:\n")

            for node_id, device in device_map.items():
                if(node_id >= width * height):
                    print(f"[警告] 节点 {node_id} 超出拓扑范围，跳过")
                    continue
                x = node_id // width
                y = node_id % width
                file.write(f"  # Process {node_id}\n")

                if device == "CPU":
                    file.write(f'  - cmd: "$SIMULATOR_ROOT/snipersim/run-sniper"\n')
                    file.write(f'    args: ["--", "$BENCHMARK_ROOT/build/bin/Transformer", "--srcX", "{x}", "--srcY", "{y}", "--topology_width", "{width}"]\n')
                    file.write(f'    log: "sniper.{x}.{y}.log"\n')
                    file.write(f'    is_to_stdout: false\n')
                    file.write(f'    clock_rate: {clock_rate_map[node_id]}\n')
                elif device == "GPU":
                    file.write(f'  - cmd: "$BENCHMARK_ROOT/build/bin/matmul_cu"\n')
                    file.write(f'    args: ["{x}", "{y}", "{x_cpu}", "{y_cpu}", "8"]\n')
                    file.write(f'    log: "gpgpusim.{x}.{y}.log"\n')
                    file.write(f'    is_to_stdout: false\n')
                    file.write(f'    clock_rate: {clock_rate_map[node_id]}\n')
                    file.write(f'    pre_copy: "$SIMULATOR_ROOT/gpgpu-sim/configs/tested-cfgs/SM7_TITANV/*"\n')

                else:
                    print(f"[警告] 不支持的设备类型: {device}，跳过节点 {node_id}")
            
            file.write("\nphase2:\n")
            file.write("  # Process 0\n")
            file.write('  - cmd: "$SIMULATOR_ROOT/popnet_chiplet/build/popnet"\n')
            file.write(f'    args: ["-A", "{width * height}", "-c", "1", "-V", "3", "-B", "12", "-O", "12", "-F", "2", "-L", "1000", "-T", "1000000000000000000000", "-r", "1", "-I", "../bench.txt", "-G", "{GV_path}", "-D", "../delayInfo.txt", "-P"]\n')
            file.write('    log: "popnet_0.log"\n')
            file.write('    is_to_stdout: false\n')
            file.write(f'    clock_rate: {popnet_clock_rate}\n')

        print(f"[完成] YML 文件已生成：{yml_path}")
    except Exception as e:
        print(f"生成YML文件时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate YML file from CSV")
    parser.add_argument("-c", "--csv_path", type=str, default="mapDevice.csv", help="Path to the CSV file")
    parser.add_argument("-y", "--yml_path", type=str, default="auto_transformer.yml", help="Path to the YML file")
    # parser.add_argument("--device_num", type=int, default=4, help="Number of devices")
    
    parser.add_argument("-w", "--width", type=int, default=2, help="topology width")
    parser.add_argument("-H", "--height", type=int, default=2, help="topology height")
    parser.add_argument("-g", "--GV_path", type=str, default=f"/home/qc/Chiplet_Heterogeneous_newVersion_gem5/Chiplet_Heterogeneous_newVersion/benchmark/auto_transformer/mesh_2_2.gv", help="Path to the GV file")
    parser.add_argument("-p", "--popnet_clock_rate", type=int, default=1, help="popnet clock rate")
    args = parser.parse_args()
    generate_yaml_file_from_csv(args.csv_path, args.yml_path, args.width, args.height, args.GV_path, args.popnet_clock_rate)
