import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd  # 导入pandas库
import numpy as np
import random
# 检查CUDA的可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 数据预处理
# transform = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
#
# # 加载数据集
# train_dataset = datasets.ImageFolder(root='./dataset/train', transform=transform)
# test_dataset = datasets.ImageFolder(root='./dataset/test', transform=transform)
#
# # 分割数据集为N个子集
# def split_dataset(dataset, num_clients):
#     dataset_length = len(dataset)
#     lengths = [dataset_length // num_clients for _ in range(num_clients)]
#     lengths[0] += dataset_length - sum(lengths)  # add any leftover to the first subset
#     subsets = random_split(dataset, lengths)
#     return subsets
#
#
# def split_dataset_iid(dataset, num_clients):
#     dataset_length = len(dataset)
#     # 打乱数据集
#     indices = list(range(dataset_length))
#     random.shuffle(indices)
#     shuffled_dataset = [dataset[i] for i in indices]
#
#     # 计算每个客户端的数据集大小
#     lengths = [dataset_length // num_clients for _ in range(num_clients)]
#     lengths[0] += dataset_length - sum(lengths)  # 将剩余的数据添加到第一个子集
#
#     # 分割数据集
#     subsets = random_split(shuffled_dataset, lengths)
#     return subsets
#
#
# def split_dataset_non_iid(dataset, num_clients):
#     dataset_length = len(dataset)
#
#     # 打乱数据集
#     indices = list(range(dataset_length))
#     random.shuffle(indices)
#     shuffled_dataset = [dataset[i] for i in indices]
#
#     # 分配数据集大小
#     lengths = [(dataset_length // num_clients+100) - (i * 2) for i in range(num_clients)]
#
#     # 确保每个客户端的数据集大小不为0
#     for i in range(len(lengths)):
#         if lengths[i] <= 0:
#             lengths[i] = 1  # 给出最小数据集大小
#
#     # 调整最后一个客户端的数据集大小以确保所有数据都被分配
#     lengths[-1] = dataset_length - sum(lengths[:-1])
#
#     # 分割数据集
#     subsets = random_split(shuffled_dataset, lengths)
#     # # 打印每个子数据集的大小
#     # for i, subset in enumerate(subsets):
#     #     print(f"客户端 {i+1} 的数据集大小: {len(subset)}")
#     return subsets
#
# # 定义客户端数量
# num_clients = 10  # 假设有N个客户端
# cluster_num = 10  # 假设有M个集群
# # 分割数据集为IID
# # all_subsets = split_dataset(train_dataset, num_clients * cluster_num)
# # 分割数据集为非IID
# all_subsets = split_dataset_non_iid(train_dataset, num_clients * cluster_num)
#
# # 为每个模型组中的每个客户端创建数据加载器
# train_loaders = [[DataLoader(subset, batch_size=512, shuffle=True) for subset in all_subsets[i*num_clients:(i+1)*num_clients]] for i in range(cluster_num)]
# test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# 定义客户端模型

# 定义客户端模型
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=43):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # Layer1
        self.layer1 = self._make_layer(block, 64, layers[0])
        # Layer2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # Layer3
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # Layer4
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        resnet = resnet18()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()
        resnet = resnet18()

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = resnet.fc

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ]
# 实例化基本的客户端模型
base_client_model = ClientModel().to(device)
base_server_model = ServerModel().to(device)

# 实例化基本的客户端模型

# 保存模型的初始状态
torch.save(base_client_model.state_dict(), 'base_model/base_client_model_state010.pth')
torch.save(base_server_model.state_dict(), 'base_model/base_server_model_state010.pth')
