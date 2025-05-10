import onnx
import json

def print_onnx_graph_info(onnx_file, output_file):
    # 加载 ONNX 模型
    model = onnx.load(onnx_file)
    graph = model.graph

    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"=== ONNX Model: {onnx_file} ===\n\n")

        # 打印模型输入
        f.write("### Model Inputs ###\n")
        for input_tensor in graph.input:
            f.write(f"Name: {input_tensor.name}, Type: {input_tensor.type}\n")
        f.write("\n")

        # 打印模型输出
        f.write("### Model Outputs ###\n")
        for output_tensor in graph.output:
            f.write(f"Name: {output_tensor.name}, Type: {output_tensor.type}\n")
        f.write("\n")

        # 打印每个节点的信息
        f.write("### Graph Nodes ###\n")
        for node in graph.node:
            f.write(f"Node: {node.name if node.name else 'Unnamed Node'}\n")
            f.write(f"  Operator Type: {node.op_type}\n")
            f.write(f"  Inputs: {', '.join(node.input)}\n")
            f.write(f"  Outputs: {', '.join(node.output)}\n")
            
            # 打印每一层的所有参数信息
            f.write("  Attributes:\n")
            for attr in node.attribute:
                attr_value = None
                if attr.type == onnx.AttributeProto.FLOAT:
                    attr_value = attr.f
                elif attr.type == onnx.AttributeProto.INT:
                    attr_value = attr.i
                elif attr.type == onnx.AttributeProto.STRING:
                    attr_value = attr.s.decode('utf-8')
                elif attr.type == onnx.AttributeProto.TENSOR:
                    attr_value = json.dumps({
                        "dims": list(attr.t.dims),
                        "data_type": attr.t.data_type
                    })
                elif attr.type == onnx.AttributeProto.INTS:
                    attr_value = list(attr.ints)
                elif attr.type == onnx.AttributeProto.FLOATS:
                    attr_value = list(attr.floats)
                elif attr.type == onnx.AttributeProto.STRINGS:
                    attr_value = [s.decode('utf-8') for s in attr.strings]
                
                f.write(f"    - {attr.name}: {attr_value}\n")
            f.write("\n")

        # 打印初始化参数（权重、常数等）
        f.write("### Initializer (Weights and Constants) ###\n")
        for initializer in graph.initializer:
            f.write(f"Name: {initializer.name}, Shape: {initializer.dims}, Data Type: {initializer.data_type}\n")
        f.write("\n")

        f.write("=== End of ONNX Model Info ===\n")

# 示例：运行该函数
onnx_file_path = "head.bbox.onnx"  # 替换成你的 ONNX 文件路径
output_file_path = "head.bbox.txt"
print_onnx_graph_info(onnx_file_path, output_file_path)

print(f"ONNX 信息已写入 {output_file_path}")