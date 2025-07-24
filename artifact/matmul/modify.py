import yaml
import sys
import os
import argparse
import re

def modify_topology_and_flit(yaml_file_path, makefile_path, new_topology_path, flit_size):
    """
    修改YAML文件中的topology路径和flit size参数
    
    Args:
        yaml_file_path: YAML文件的路径
        new_topology_path: 新的topology文件路径
        flit_size: 新的flit size值
    """
    try:
        # 检查YAML文件是否存在
        if not os.path.exists(yaml_file_path):
            print(f"错误: YAML文件不存在: {yaml_file_path}")
            return False
        # 检查makefile是否存在
        if not os.path.exists(makefile_path):   
            print(f"错误: Makefile不存在: {makefile_path}")
            return False
            
        # 读取YAML文件内容
        with open(yaml_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        with open(makefile_path, 'r', encoding='utf-8') as file:
            makefile_content = file.read()
        
        file_pattern = r'(-f\s+)(\S+)'
        file_match = re.search(file_pattern, makefile_content)
        if file_match:
            old_flit_size = file_match.group(2)
            new_flit_size = str(flit_size)
            makefile_content = makefile_content.replace(old_flit_size, new_flit_size)
            print(f"成功修改Makefile中的flit size:")
            print(f"  原flit size: {old_flit_size}")
            print(f"  新flit size: {new_flit_size}")
        else:
            print("警告: 未找到Makefile中的flit size参数")
        # 修改topology路径 - 查找并替换 "../topology/xxx.gv" 模式
        topology_pattern = r'(../topology/[^"]+\.gv)'
        topology_match = re.search(topology_pattern, content)
        if topology_match:
            old_topology = topology_match.group(1)
            content = content.replace(old_topology, new_topology_path)
            print(f"成功修改topology路径:")
            print(f"  原路径: {old_topology}")
            print(f"  新路径: {new_topology_path}")
        else:
            print("警告: 未找到topology路径模式")
        
        # 修改flit size参数 - 查找并替换 "-F", "数字" 模式
        flit_pattern = r'("-F",\s*")\d+(")'
        flit_match = re.search(flit_pattern, content)
        if flit_match:
            old_flit_part = flit_match.group(0)
            new_flit_part = f'"-F", "{flit_size}"'
            content = content.replace(old_flit_part, new_flit_part)
            print(f"成功修改flit size参数:")
            print(f"  原参数: {old_flit_part}")
            print(f"  新参数: {new_flit_part}")
        else:
            print("警告: 未找到-F参数")
        
        # 写回文件，保持原有格式
        with open(yaml_file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        
        with open(makefile_path, 'w', encoding='utf-8') as file:
            file.write(makefile_content)
        
        print(f"已成功保存修改到文件: {yaml_file_path}")
        return True
        
    except FileNotFoundError:
        print(f"错误: 文件未找到: {yaml_file_path}")
        return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def main():
    """
    主函数，处理命令行参数
    """
    parser = argparse.ArgumentParser(description="Modify topology path and flit size in YAML file")
    parser.add_argument("--topology", default="mesh", help="Type of topology to modify")
    parser.add_argument("--flit_size", type=int, default=2, help="Flit size in bytes")
    parser.add_argument("--width", type=int, default=6, help="Topology width")
    parser.add_argument("--height", type=int, default=6, help="Topology height")
    
    args = parser.parse_args()
    
    # 构建新的topology路径
    new_topology_path = f"../topology/{args.topology}_{args.width}_{args.height}_flit_{args.flit_size}.gv"
    yaml_file_path = "./matmul.yml"
    makefile_path = "./makefile"
    
    # 转换为绝对路径以便调试
    yaml_file_path = os.path.abspath(yaml_file_path)
    makefile_path = os.path.abspath(makefile_path)
    
    print(f"准备修改配置:")
    print(f"  YAML文件: {yaml_file_path}")
    print(f"  Makefile: {makefile_path}")
    print(f"  新topology路径: {new_topology_path}")
    print(f"  新flit size: {args.flit_size}")
    print("-" * 50)
    
    # 执行修改
    success = modify_topology_and_flit(yaml_file_path, makefile_path, new_topology_path, args.flit_size)
    
    if success:
        print("操作完成！")
        sys.exit(0)
    else:
        print("操作失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()