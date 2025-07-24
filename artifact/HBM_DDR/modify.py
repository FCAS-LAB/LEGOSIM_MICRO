import yaml
import sys
import os
import argparse
import re

def modify_topology_and_flit(yaml_file_path, storage_type):
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
            
        # 读取YAML文件内容
        with open(yaml_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        file_pattern = r'(DDR|HBM)'
        file_match = re.search(file_pattern, content)
        if file_match:
            old_storage_type = file_match.group(0)
            new_storage_type = storage_type
            content = content.replace(old_storage_type, new_storage_type)
            print(f"成功修改Makefile中的flit size:")
            print(f"  原 storage type: {old_storage_type}")
            print(f"  新 storage type: {new_storage_type}")
        else:
            print("警告: 未找到Makefile中的flit size参数")
        
        
        # 写回文件，保持原有格式
        with open(yaml_file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        
        
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
    parser.add_argument("--type", default="DDR", help="Type of topology to modify")
    
    args = parser.parse_args()
    
    # 构建新的topology路径
    yaml_file_path = "./storage.yml"
    
    # 转换为绝对路径以便调试
    yaml_file_path = os.path.abspath(yaml_file_path)
    
    print(f"准备修改配置:")
    print(f"  YAML文件: {yaml_file_path}")
    print("-" * 50)
    
    # 执行修改
    success = modify_topology_and_flit(yaml_file_path, args.type)
    
    if success:
        print("操作完成！")
        sys.exit(0)
    else:
        print("操作失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()