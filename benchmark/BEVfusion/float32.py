# import onnx
# import numpy as np

# # 加载原始模型
# model = onnx.load('head.bbox.onnx')

# # 将所有 float16 转换为 float32
# for initializer in model.graph.initializer:
#     if initializer.data_type == 10:  # 10 是 float16 的类型标识
#         # 获取数据
#         data = onnx.numpy_helper.to_array(initializer)
#         # 转换为 float32
#         data = data.astype(np.float32)
#         # 创建新的初始化器
#         new_initializer = onnx.numpy_helper.from_array(data, initializer.name)
#         # 替换原始初始化器
#         initializer.CopyFrom(new_initializer)

# # 修改所有输入输出的数据类型
# for input in model.graph.input:
#     if input.type.tensor_type.elem_type == 10:
#         input.type.tensor_type.elem_type = 1

# for output in model.graph.output:
#     if output.type.tensor_type.elem_type == 10:
#         output.type.tensor_type.elem_type = 1

# # 保存转换后的模型
# onnx.save(model, 'head.bbox.float32.onnx')

import onnx
import numpy as np

def print_tensor_info(model):
    print("Initializers:")
    for initializer in model.graph.initializer:
        print(f"Name: {initializer.name}")
        print(f"Data type: {initializer.data_type}")
        print("---")
    
    print("\nInputs:")
    for input in model.graph.input:
        print(f"Name: {input.name}")
        print(f"Data type: {input.type.tensor_type.elem_type}")
        print("---")
    
    print("\nOutputs:")
    for output in model.graph.output:
        print(f"Name: {output.name}")
        print(f"Data type: {output.type.tensor_type.elem_type}")
        print("---")

# 加载原始模型
model = onnx.load('lidar.backbone.xyz.onnx')

# 打印转换前的信息
print("Before conversion:")
print_tensor_info(model)

# 将所有不支持的数据类型转换为 float32
for initializer in model.graph.initializer:
    if initializer.data_type != 1:  # 1 是 float32 的类型标识
        print(f"Converting {initializer.name} from type {initializer.data_type} to float32")
        # 获取数据
        data = onnx.numpy_helper.to_array(initializer)
        # 转换为 float32
        data = data.astype(np.float32)
        # 创建新的初始化器
        new_initializer = onnx.numpy_helper.from_array(data, initializer.name)
        # 替换原始初始化器
        initializer.CopyFrom(new_initializer)

# 修改所有输入输出的数据类型
for input in model.graph.input:
    if input.type.tensor_type.elem_type != 1:
        print(f"Converting input {input.name} to float32")
        input.type.tensor_type.elem_type = 1

for output in model.graph.output:
    if output.type.tensor_type.elem_type != 1:
        print(f"Converting output {output.name} to float32")
        output.type.tensor_type.elem_type = 1

# 打印转换后的信息
print("\nAfter conversion:")
print_tensor_info(model)

# 保存转换后的模型
onnx.save(model, 'lidar.backbone.xyz.float32.onnx')
print("\nModel saved as lidar.backbone.xyz.float32.onnx")