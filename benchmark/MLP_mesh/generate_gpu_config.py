#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def generate_yaml_file():
    with open('mlp_config.yml', 'w') as file:
        # 添加注释和头部
        file.write("# 自动生成的配置文件 - 包含50个GPU芯粒和其他处理器\n")
        file.write("# Phase 1 configuration.\n")
        file.write("phase1:\n")
        
        # 第一组GPU芯粒 (0,0)-(4,4)
        for i in range(5):
            for j in range(5):
                file.write(f"  # Process {i*5+j}\n")
                file.write(f'  - cmd: "$BENCHMARK_ROOT/bin/mlp_cu"\n')
                file.write(f'    args: ["{i}", "{j}"]\n')
                file.write(f'    log: "gpgpusim.{i}.{j}.log"\n')
                file.write(f'    is_to_stdout: false\n')
                file.write(f'    clock_rate: 1\n')
                file.write(f'    pre_copy: "$SIMULATOR_ROOT/gpgpu-sim/configs/tested-cfgs/SM2_GTX480/*"\n')
        
        # 第二组GPU芯粒 (0,5)-(4,9)
        for i in range(5):
            for j in range(5, 10):
                process_num = i*5+(j-5)+25
                file.write(f"  # Process {process_num}\n")
                file.write(f'  - cmd: "$BENCHMARK_ROOT/bin/mlp_cu"\n')
                file.write(f'    args: ["{i}", "{j}"]\n')
                file.write(f'    log: "gpgpusim.{i}.{j}.log"\n')
                file.write(f'    is_to_stdout: false\n')
                file.write(f'    clock_rate: 1\n')
                file.write(f'    pre_copy: "$SIMULATOR_ROOT/gpgpu-sim/configs/tested-cfgs/SM2_GTX480/*"\n')

        # Sniper处理器
        file.write(f"  # Process 50\n")
        file.write(f'  - cmd: "$SIMULATOR_ROOT/snipersim/run-sniper"\n')
        file.write(f'    args: ["--", "$BENCHMARK_ROOT/bin/mlp_c", "9", "9"]\n')
        file.write(f'    log: "sniper.9.9.log"\n')
        file.write(f'    is_to_stdout: false\n')
        file.write(f'    clock_rate: 1\n')

        # CIM处理器
        file.write(f"  # Process 51\n")
        file.write(f'  - cmd: "$BENCHMARK_ROOT/bin/cim"\n')
        file.write(f'    args: ["9", "8"]\n')
        file.write(f'    log: "npu.9.8.log"\n')
        file.write(f'    is_to_stdout: false\n')
        file.write(f'    clock_rate: 1\n')

        # MNSIM处理器
        file.write(f"  # Process 52\n")
        file.write(f'  - cmd: "$BENCHMARK_ROOT/bin/mnsim"\n')
        file.write(f'    args: ["9", "7"]\n')
        file.write(f'    log: "mnsim.9.7.log"\n')
        file.write(f'    is_to_stdout: false\n')
        file.write(f'    clock_rate: 1\n')

        # Phase 2 配置
        file.write("# Phase 2 configuration.\n")
        file.write("phase2:\n")
        file.write("  # Process 0\n")
        file.write('  - cmd: "$SIMULATOR_ROOT/popnet_chiplet/build/popnet"\n')
        file.write('    args: ["-A", "100", "-c", "1", "-V", "3", "-B", "12", "-O", "12", "-F", "2", "-L", "1000", "-T", "1000000000", "-r", "1", "-I", "../bench.txt", "-G", "test/mesh_6_6.gv", "-D", "../delayInfo.txt", "-P"]\n')
        file.write('    log: "popnet_0.log"\n')
        file.write('    is_to_stdout: false\n')
        file.write('    clock_rate: 1\n')

        # 文件配置
        file.write("\n# File configuration. (Not used yet)\n")
        file.write('bench_file: "./bench_torus_flit_4.txt"\n')
        file.write('delayinfo_file: "./delayInfo_torus_flit_4.txt"\n')
    
    print("配置文件已生成: mlp_config.yml")

if __name__ == "__main__":
    generate_yaml_file()