import onnx
import numpy as np
from onnx import numpy_helper

def replace_negative_one_in_reshape(model_path, output_path):
    # 加载 ONNX 模型
    model = onnx.load(model_path)
    nodes_with_issues = []  # 用于记录无法推断的节点，记录为 (node_name, shape_values 或 None)

    # 遍历所有节点
    for node in model.graph.node:
        if node.op_type == "Reshape":
            shape_replaced = False  # 标志是否替换了目标形状中的 `-1`

            # 查找与 shape 相关的 initializer
            for input_name in node.input:
                for initializer in model.graph.initializer:
                    if initializer.name == input_name:
                        shape_values = numpy_helper.to_array(initializer)
                        
                        # 确保 shape 是整数数组
                        if not isinstance(shape_values, np.ndarray) or shape_values.dtype.kind != 'i':
                            continue

                        # 检查是否存在 -1
                        if -1 in shape_values:
                            # 获取输入张量的信息
                            input_tensor_name = node.input[0]  # 输入数据的名称
                            input_tensor = next((x for x in model.graph.value_info if x.name == input_tensor_name), None)
                            
                            # 如果输入张量形状不完整，则无法推断
                            if input_tensor is None or len(input_tensor.type.tensor_type.shape.dim) == 0:
                                nodes_with_issues.append((node.name, shape_values))
                                continue

                            # 获取输入张量的总元素数
                            input_shape = [
                                dim.dim_value if dim.dim_value > 0 else None
                                for dim in input_tensor.type.tensor_type.shape.dim
                            ]

                            if None in input_shape:
                                nodes_with_issues.append((node.name, shape_values))
                                continue

                            input_total_elements = np.prod(input_shape)
                            known_dims_product = np.prod(
                                [dim for dim in shape_values if dim > 0]
                            )

                            # 如果无法推断唯一的 -1 值
                            if input_total_elements % known_dims_product != 0:
                                nodes_with_issues.append((node.name, shape_values))
                                continue

                            # 计算 -1 的值
                            negative_one_value = input_total_elements // known_dims_product
                            shape_values = np.where(shape_values == -1, negative_one_value, shape_values)

                            # 替换 initializer 的值
                            new_initializer = numpy_helper.from_array(shape_values, initializer.name)
                            model.graph.initializer.remove(initializer)
                            model.graph.initializer.append(new_initializer)
                            shape_replaced = True

            # 如果没有替换，则记录为有问题的节点
            if not shape_replaced:
                nodes_with_issues.append((node.name, None))

    # 保存修改后的模型
    onnx.save(model, output_path)

    # 打印有问题的节点
    if nodes_with_issues:
        print("Nodes with issues (cannot replace -1):")
        for node in nodes_with_issues:
            if node[1] is not None:
                print(f"Node: {node[0]}, Shape: {node[1]}")
            else:
                print(f"Node: {node[0]} (Shape could not be retrieved)")
    else:
        print("All -1 values were successfully replaced.")

if __name__ == "__main__":
    # 替换为你的模型路径
    input_model_path = "camera.backbone.onnx"
    output_model_path = "camera.backbone_2.onnx"
    replace_negative_one_in_reshape(input_model_path, output_model_path)
