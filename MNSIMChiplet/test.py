import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# # 定义输入矩阵
# input_matrix = torch.randn(1, 1, 5, 5)  # 假设输入矩阵大小为 1x1x5x5

# # 定义卷积层
# conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)

# # 打印卷积核权重
# print("卷积核权重：", conv_layer.weight)

# # 运行卷积层
# output_matrix = conv_layer(input_matrix)

# # 输出结果
# print("输出矩阵大小：", output_matrix.size())
# print("输出矩阵：", output_matrix)
# model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# model = models.resnet18()
# model = models.resnet18(pretrained=False)
state_dict = torch.load('/root/f/MNSIM-2.0/cifar10_resnet18_params.pth',  map_location='cuda:0')

for item in state_dict:
    print(item)
    # print(model.state_dict()[item].size())

# feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # 提取特征的部分
# classifier = model.fc

# print(list(model.children())[5])
# print(len(list(model.children())))
# print(feature_extractor)
# print(classifier)


# model_children = list(model.children())

# # 划分成八个部分
# num_sections = 8
# section_size = len(model_children) // num_sections

# sections = [model_children[i*section_size:(i+1)*section_size] for i in range(num_sections)]

# # 打印每个部分的信息
# for i, section in enumerate(sections):
#     print(f"Section {i+1}: {section}")