import onnx
import numpy as np
from onnx import numpy_helper

def find_reshape_with_multiple_negative_ones(model_path):
    # 加载 ONNX 模型
    model = onnx.load(model_path)

    # 遍历所有节点
    for node in model.graph.node:
        if node.op_type == "Reshape":
            for input_name in node.input:
                # 检查输入中与 shape 相关的 constant 值
                for initializer in model.graph.initializer:
                    if initializer.name == input_name:
                        # 获取 shape 值
                        shape_values = numpy_helper.to_array(initializer)
                        # 检查是否存在两个或多个 -1
                        negative_one_count = np.sum(shape_values == -1)
                        if negative_one_count >= 2:
                            print(f"Found 'Reshape' node with multiple -1s in shape: {node.name}")
                            print(f"Shape values: {shape_values}")
                            break

if __name__ == "__main__":
    # 替换为你的模型路径
    model_path = "camera.backbone_2.onnx"
    find_reshape_with_multiple_negative_ones(model_path)
