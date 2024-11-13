import sys 

def read_formatted_txt(file_path="bench.txt"):
    with open(file_path, 'r') as file:
        data = [line.split() for line in file.readlines()]
    return data
 
# 使用函数
# formatted_data = read_formatted_txt('bench.txt')
# print(formatted_data)