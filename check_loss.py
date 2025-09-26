import torch
from torch.utils.data import DataLoader
from train_eval import PrecomputedDataset, ResNet50

# 加载预生成数据
precomputed_dir = "./precomputed_data/sd10/"
dataset = PrecomputedDataset(precomputed_dir)
dataloader = DataLoader(dataset, batch_size=4, num_workers=0)

# 初始化模型（根据数据形状修正channel参数）
model = ResNet50(num_zernike=10, channel=3)  # 数据是3通道，所以channel=3
model.eval()

# 获取一批数据
data, target = next(iter(dataloader))
print("数据形状:", data.shape)
print("标签形状:", target.shape)

# 前向传播
output = model(data)
print("模型输出形状:", output.shape)

# 计算损失
criterion = torch.nn.MSELoss()
loss = criterion(output, target)
print("初始损失:", loss.item())