import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd  # 导入pandas库
import numpy as np
import random
#from sklearn.model_selection import StratifiedShuffleSplit
for j in range(1,11):
    # 检查CUDA的可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载CIFAR10数据集
    train_dataset = datasets.CIFAR10(root='./dataset_mnist/train', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./dataset_mnist/test', train=False, download=True, transform=transform)


    # 分割数据集为N个子集
    def split_dataset(dataset, num_clients, seed=j):
        # 设置随机数种子以确保结果的可重复性
        torch.manual_seed(seed)

        dataset_length = len(dataset)
        lengths = [dataset_length // num_clients for _ in range(num_clients)]
        lengths[0] += dataset_length - sum(lengths)  # 将剩余部分加到第一个子集中

        # 使用固定的种子进行数据集划分
        subsets = random_split(dataset, lengths)
        return subsets
    # def split_dataset_non_iid_fixed(dataset_cifar10, num_clients,):#Propose1不能收敛的版本
    #     dataset_length = len(dataset_cifar10)
    #     lengths = [dataset_length // num_clients for _ in range(num_clients)]
    #     lengths[0] += dataset_length - sum(lengths)  # 将剩余部分加到第一个子集
    #
    #     subsets = []
    #     start = 0
    #     for length in lengths:
    #         end = start + length
    #         subsets.append(Subset(dataset_cifar10, range(start, end)))
    #         start = end
    #
    #     return subsets
    # def split_dataset_non_iid_fixed(dataset_cifar10, num_clients, seed=42):# 数据集数量对不上的版本
    #     random.seed(seed)
    #
    #     classes = dataset_cifar10.classes
    #     class_to_indices = {cls: [] for cls in classes}
    #     for idx, (_, class_idx) in enumerate(dataset_cifar10.imgs):
    #         cls = classes[class_idx]
    #         class_to_indices[cls].append(idx)
    #
    #     # 确保每个客户端获得至少一个类别
    #     all_classes = list(classes)
    #     random.shuffle(all_classes)
    #     clients_classes = [set() for _ in range(num_clients)]
    #     for i, cls in enumerate(all_classes):
    #         clients_classes[i % num_clients].add(cls)
    #
    #     # 随机分配剩余的类别
    #     for _ in range(num_clients):
    #         for cls in random.sample(all_classes, k=len(all_classes)):
    #             client_idx = random.randrange(num_clients)
    #             clients_classes[client_idx].add(cls)
    #
    #     # 为每个客户端创建数据子集
    #     print(f"总的的数据集大小: {len(dataset_cifar10)}")
    #     subsets = []
    #     for client_classes in clients_classes:
    #         indices = []
    #         for cls in client_classes:
    #             indices.extend(class_to_indices[cls])
    #         if indices:  # 确保子集不为空
    #             random.shuffle(indices)
    #             subsets.append(Subset(dataset_cifar10, indices))
    #     # 打印每个客户端的数据集大小
    #     for i, subset in enumerate(subsets):
    #         print(f"客户端 {i+1} 的数据集大小: {len(subset)}")
    #
    #     return subsets#￥
    def split_dataset_non_iid_fixed(dataset, num_clients, seed=42):
        random.seed(seed)


        # 定义每个用户组分配的类别数
        category_distribution = [2, 2,2, 2,2,2,2,2,2,2,2, 2,2, 2,2,2,2,2,2,5]

        # 计算每个类别的大小
        category_sizes = {}
        for _, label in dataset:
            if label in category_sizes:
                category_sizes[label] += 1
            else:
                category_sizes[label] = 1

        # 确保类别数量足够分配
        assert sum(category_distribution) <= len(category_sizes), "类别数不足以分配给所有用户组"

        # 分配样本到不同的用户组
        category_indices = sorted(list(category_sizes.keys()))
        samples_per_group = []
        start_idx = 0
        for group in category_distribution:
            group_total = sum([category_sizes[category_indices[i]] for i in range(start_idx, start_idx + group)])
            start_idx += group
            samples_per_group.append(group_total)

        # 计算每个用户的样本数
        lengths = []
        for group_total in samples_per_group:
            per_client = group_total // (num_clients // len(category_distribution))
            lengths.extend([per_client] * (num_clients // len(category_distribution)))

        # 调整长度以确保总和与数据集大小匹配
        lengths[-1] += len(dataset) - sum(lengths)

        subsets = random_split(dataset, lengths)
        # 打印每个子数据集的大小
        for i, subset in enumerate(subsets):
            print(f"客户端 {i+1} 的数据集大小: {len(subset)}")
        return subsets


    def split_dataset_iid(dataset, num_clients):
        dataset_length = len(dataset)
        # 打乱数据集
        indices = list(range(dataset_length))
        random.shuffle(indices)
        shuffled_dataset = [dataset[i] for i in indices]

        # 计算每个客户端的数据集大小
        lengths = [dataset_length // num_clients for _ in range(num_clients)]
        lengths[0] += dataset_length - sum(lengths)  # 将剩余的数据添加到第一个子集

        # 分割数据集
        subsets = random_split(shuffled_dataset, lengths)
        for i, subset in enumerate(subsets):
            print(f"客户端 {i+1} 的数据集大小: {len(subset)}")
        return subsets


    def split_dataset_non_iid(dataset, Total_clients, alpha, seed=j):
        # 设置随机数种子以确保结果的可重复性
        np.random.seed(seed)

        # 假设数据集已经按照类别标签进行排序
        n_classes = len(np.unique([label for _, label in dataset]))
        n_samples_per_class = np.bincount([label for _, label in dataset], minlength=n_classes)

        # 按类别分配给每个客户端
        client_indices = [[] for _ in range(Total_clients)]
        for c in range(n_classes):
            # 按迪利克雷分布生成这一类的分配比例
            proportions = np.random.dirichlet(np.repeat(alpha, Total_clients))

            # 根据比例和该类的样本数量计算每个客户端的样本数
            samples_per_client = (proportions * n_samples_per_class[c]).astype(int)

            # 确保总数正确
            samples_per_client[-1] = n_samples_per_class[c] - samples_per_client[:-1].sum()

            # 分配样本
            shuffled_indices = np.random.permutation(np.where(np.array([label for _, label in dataset]) == c)[0])
            for i in range(Total_clients):
                client_indices[i].extend(shuffled_indices[:samples_per_client[i]].tolist())
                shuffled_indices = shuffled_indices[samples_per_client[i]:]

        # 使用Subset创建子数据集
        subsets = [Subset(dataset, indices) for indices in client_indices]
        return subsets
    # 定义客户端数量
    num_clients = 6  # 假设有N个客户端
    Total_clients=30
    cluster_num = 5 # 假设有M个集群
    batch_size=64

    # 分割数据集为IID
    # all_subsets = split_dataset(train_dataset, num_clients * cluster_num)
    # 分割数据集为非IID
    all_subsets = split_dataset_non_iid(train_dataset, Total_clients,0.5)

    # 为每个模型组中的每个客户端创建数据加载器
    train_loaders = [[DataLoader(subset, batch_size, shuffle=True) for subset in all_subsets[i*num_clients:(i+1)*num_clients]] for i in range(cluster_num)]
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

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




    # 实例化三组模型
    # models_groups = [
    #     {
    #         "client_models": [ClientModel().to(device) for _ in range(num_clients)],
    #         "server_model": ServerModel().to(device),  # 为每个模型组创建一个服务器模型
    #         "client_optimizers": [optim.Adam(client_model.parameters(), lr=0.001) for client_model in [ClientModel().to(device) for _ in range(num_clients)]],
    #         "last_client_model_state": None,
    #         "prev_client_model_state": None
    #     } for _ in range(cluster_num)
    # ]
    # 实例化基本的客户端模型
    base_client_model_path = f'base_model/base_client_model_state0{j}.pth' if j < 11 else f'base_model/base_client_model_state{j}.pth'
    base_client_model = torch.load(base_client_model_path)

    base_server_model_path = f'base_model/base_server_model_state0{j}.pth' if j < 11 else f'base_model/base_server_model_state{j}.pth'
    base_server_model = torch.load(base_server_model_path)
    # 实例化三组模型
    models_groups = []

    for _ in range(cluster_num):
        # 为每个客户端创建一个模型副本
        client_models = []
        for _ in range(num_clients):
            client_model = ClientModel().to(device)
            client_model.load_state_dict(base_client_model)
            client_models.append(client_model)

        # 创建对应的客户端优化器
        client_optimizers = [optim.Adam(client_model.parameters(), lr=0.01) for client_model in client_models]

        # 为每个模型组创建一个服务器模型
        server_model = ServerModel().to(device)
        server_model.load_state_dict(base_server_model)
        # 将模型和优化器添加到模型组中
        model_group = {
            "client_models": client_models,
            "server_model": server_model,
            "client_optimizers": client_optimizers,
            "last_client_model_state": None,
            "prev_client_model_state": None
        }

        models_groups.append(model_group)



    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练和测试函数保持不变
    # 训练过程
    def train(client_model, server_model, train_loader, client_optimizer, server_optimizer):
        client_model.train()
        server_model.train()
        data, target = next(iter(train_loader))  # 取出一个batch的数据
        data, target = data.to(device), target.to(device)
        client_optimizer.zero_grad()
        server_optimizer.zero_grad()

        # 客户端前向传播
        client_output = client_model(data)

        # 将输出发送到服务器（模拟）
        server_input = client_output.detach().requires_grad_()

        # 服务器端前向传播和反向传播
        server_output = server_model(server_input)
        loss = criterion(server_output, target)
        loss.backward()

        # 更新服务器和客户端模型
        server_optimizer.step()
        client_output.backward(server_input.grad)
        client_optimizer.step()
        return loss

    # 测试过程
    def test(client_model, server_model, test_loader):
        client_model.eval()
        server_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                client_output = client_model(data)
                server_output = server_model(client_output)
                _, predicted = torch.max(server_output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy on the test set: {accuracy:.2f}%')
        return accuracy

    def federated_average(models):
        """
        Perform federated averaging on the models.
        """
        model_state_dicts = [model.state_dict() for model in models]
        averaged_state_dict = {}

        for key in model_state_dicts[0].keys():
            averaged_state_dict[key] = sum([model_state_dict[key] for model_state_dict in model_state_dicts]) / len(model_state_dicts)

        return averaged_state_dict




    # 串行训练*******************************************
    num_rounds = 300
    # 创建M个服务器优化器
    server_optimizers = [optim.Adam(models_groups[i]["server_model"].parameters(), lr=0.001) for i in range(cluster_num)]
    # 初始化空的DataFrame用于存储结果
    results_df = pd.DataFrame(columns=['Num Clients', 'Cluster Num', 'Round', 'Accuracy'])

    for round in range(num_rounds):
        print(f"Round {round + 1}")

        for idx, model_group in enumerate(models_groups):
            client_models = model_group["client_models"]
            server_model = model_group["server_model"]
            client_optimizers = model_group["client_optimizers"]
            server_optimizer = server_optimizers[idx]  # 使用对应的服务器优化器

            # 使用当前模型组的状态
            last_client_model_state = model_group["last_client_model_state"]
            prev_client_model_state = model_group["prev_client_model_state"]

            # 选取当前模型组的数据加载器
            current_group_train_loaders = train_loaders[idx]

            # 如果不是第一个round且last_client_model_state不为None，用上一个round的最后一个客户端的模型状态初始化第一个客户端的模型
            if last_client_model_state is not None:
                client_models[0].load_state_dict(last_client_model_state)

            # for i, (client_model, train_loader, client_optimizer) in enumerate(
            #         zip(client_models, train_loaders, client_optimizers)):
            for i, (client_model, client_optimizer) in enumerate(
                    zip(client_models, client_optimizers)):
                train_loader = current_group_train_loaders[i]
                # 如果不是第一个客户端，使用上一个客户端的模型状态
                if i > 0:
                    client_model.load_state_dict(prev_client_model_state)

                train(client_model, server_model, train_loader, client_optimizer, server_optimizer)

                # 保存当前客户端模型的状态
                prev_client_model_state = client_model.state_dict()

                # 如果是round的最后一个客户端，保存其模型状态
                if i == len(client_models) - 1:
                    last_client_model_state = client_model.state_dict()
                # 更新当前模型组的状态
                model_group["last_client_model_state"] = last_client_model_state
                model_group["prev_client_model_state"] = prev_client_model_state

        # 在这里进行模型聚合
        # 服务器模型联邦平均
        server_models = [model_group["server_model"] for model_group in models_groups]
        averaged_server_state_dict = federated_average(server_models)

        # 更新每组的服务器模型
        for model_group in models_groups:
            model_group["server_model"].load_state_dict(averaged_server_state_dict)

        # 客户端模型联邦平均
        last_client_models = [model_group["client_models"][-1] for model_group in models_groups]
        averaged_client_state_dict = federated_average(last_client_models)

        # 仅更新model_group中的last_client_model_state
        for model_group in models_groups:
            model_group["last_client_model_state"] = averaged_client_state_dict

        #测试准确率

        # 创建服务器和客户端模型的新实例
        aggregated_server_model = ServerModel().to(device)
        aggregated_client_model = ClientModel().to(device)

        # 加载聚合后的状态字典
        aggregated_server_model.load_state_dict(averaged_server_state_dict)
        aggregated_client_model.load_state_dict(averaged_client_state_dict)

        # 使用聚合后的模型进行测试
        print("Testing aggregated server model:")
        accuracy = test(aggregated_client_model, aggregated_server_model, test_loader)

        # 将每轮的结果添加到DataFrame
        new_row = pd.DataFrame([{
            'Num Clients': num_clients,
            'Cluster Num': cluster_num,
            'Round': round + 1,
            'Accuracy': accuracy
        }])
        results_df = pd.concat([results_df, new_row], ignore_index=True)

        # # 如果准确率达到94%，停止循环
        # if accuracy >= 95:
        #     print(f"Reached 94% accuracy at Round {round + 1}. Stopping training.")
        #     break
    # 在结束时保存DataFrame到Excel文件
    results_path = f'./HSFL_noniid0{j}.xlsx' if j < 11 else f'./HSFL_noniid{j}.xlsx'
    results_df.to_excel(results_path, index=False)

