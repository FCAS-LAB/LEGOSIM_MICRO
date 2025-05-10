import onnx
from onnx import helper, shape_inference

def remove_add_node(model_path, output_path):
    # 加载模型
    model = onnx.load(model_path)
    
    # 创建新的节点列表，排除Add_127
    new_nodes = []
    value_infos = list(model.graph.value_info)
    
    # 添加Resize节点的维度信息
    # 第一个Resize节点 (Resize_8894)
    resize1_input_info = helper.make_tensor_value_info(
        '11219',  # 实际的输入名称
        onnx.TensorProto.FLOAT,
        [6, 768, 8, 22]
    )
    resize1_output_info = helper.make_tensor_value_info(
        '11236',  # 实际的输出名称
        onnx.TensorProto.FLOAT,
        [6, 768, 16, 44]
    )
    
    # 第二个Resize节点 (Resize_8906)
    resize2_input_info = helper.make_tensor_value_info(
        '11245',  # 实际的输入名称
        onnx.TensorProto.FLOAT,
        [6, 256, 16, 44]
    )
    resize2_output_info = helper.make_tensor_value_info(
        '11262',  # 实际的输出名称
        onnx.TensorProto.FLOAT,
        [6, 256, 32, 88]
    )
    
    # 添加scale info
    scale1_info = helper.make_tensor_value_info(
        '11233',  # 第一个resize的scale输入
        onnx.TensorProto.FLOAT,
        [4]
    )
    scale2_info = helper.make_tensor_value_info(
        '11302',  # 第二个resize的scale输入
        onnx.TensorProto.FLOAT,
        [4]
    )
    
    value_infos.extend([
        resize1_input_info,
        resize1_output_info,
        resize2_input_info,
        resize2_output_info,
        scale1_info,
        scale2_info
    ])
    
    # 遍历原始节点，保持Resize节点的所有属性
    for node in model.graph.node:
        if node.name == 'Add_127':
            continue  # 跳过Add节点
        else:
            if node.op_type == 'Resize':
                # 确保保留所有Resize节点的属性
                new_node = helper.make_node(
                    'Resize',
                    inputs=node.input,
                    outputs=node.output,
                    name=node.name,
                    coordinate_transformation_mode='pytorch_half_pixel',
                    mode='linear',
                    nearest_mode='floor',
                    cubic_coeff_a=-0.75
                )
                new_nodes.append(new_node)
            else:
                new_nodes.append(node)
    
    # 创建新的图
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=model.graph.initializer,
        value_info=value_infos
    )
    
    # 创建新的模型
    new_model = helper.make_model(
        new_graph,
        producer_name='modified_model',
        opset_imports=[helper.make_opsetid("", 13)]
    )
    
    # 保存修改后的模型
    onnx.save(new_model, output_path)
    print(f"Model saved to {output_path} with Add_127 node removed")

if __name__ == "__main__":
    remove_add_node("camera.backbone_fixed.onnx", "camera.backbone_fixed2.onnx")