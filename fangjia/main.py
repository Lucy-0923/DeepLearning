import csv
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def processdate(date):
    """将 'YYYYMMDD...' 格式的日期转换为相对月份数（以 2014年5月 为基准）"""
    date_num = (int(date[:4]) - 2014) * 12 + (int(date[4:6]) - 5)
    return date_num


# ──────────────────────────────────────────────
# 数据集
# ──────────────────────────────────────────────

class HouseDataset(Dataset):
    def __init__(self, path, mode="train"):
        with open(path, 'r') as f:
            csv_data = list(csv.reader(f))

        # 预处理：日期数值化、价格字符串转浮点
        for index in range(1, len(csv_data)):
            csv_data[index][1] = processdate(csv_data[index][1])
            csv_data[index][2] = str(eval(csv_data[index][2]))

        data = np.array(csv_data)[1:].astype(float)

        # 删除 id(0)、price(2)、zipcode(16) 列，剩余列作为特征
        x = np.delete(data, [0, 2, 16], axis=1)
        # 价格单位换算为百万（缩小数值范围，有利于训练）
        y = data[:, 2] / 1e6

        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

        # Z-score 标准化：使每个特征均值为 0、标准差为 1
        self.x = (self.x - self.x.mean(dim=0, keepdim=True)) / self.x.std(dim=0, keepdim=True)

        print(f'Finished reading the {mode} set of House Dataset ({len(self.x)} samples found)')

    def __getitem__(self, item):
        return self.x[item].float(), self.y[item].float()

    def __len__(self):
        return len(self.x)


# ──────────────────────────────────────────────
# 模型
# ──────────────────────────────────────────────

class MyNet(nn.Module):
    def __init__(self, in_dim):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        # 若 batch 维度存在则 squeeze，使输出 shape 与 target 一致
        return x.squeeze(1) if x.dim() > 1 else x


# ──────────────────────────────────────────────
# 损失函数（MSE + L2 正则化）
# ──────────────────────────────────────────────

def mse_loss(pred, target, model, l2_lambda=0.00075):
    """MSE 损失 + L2 权重正则化，防止过拟合"""
    mse = nn.MSELoss(reduction='mean')(pred, target)
    l2_reg = sum(torch.sum(p ** 2) for p in model.parameters())
    return mse + l2_lambda * l2_reg


# ──────────────────────────────────────────────
# 训练 & 验证
# ──────────────────────────────────────────────

def train_val(model, trainloader, valloader, optimizer, loss_fn, epochs, device, save_path):
    model = model.to(device)
    plt_train_loss = []
    plt_val_loss = []
    min_val_loss = float('inf')  # 记录历史最低验证损失，用于保存最优模型

    for epoch in range(epochs):
        start_time = time.time()

        # ── 训练阶段 ──
        model.train()
        train_loss = 0.0
        for x, target in trainloader:
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(x)
            bat_loss = loss_fn(pred, target, model)
            bat_loss.backward()
            optimizer.step()
            train_loss += bat_loss.item()

        # 记录每 epoch 的平均训练损失
        plt_train_loss.append(train_loss / len(trainloader.dataset))

        # ── 验证阶段 ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, target in valloader:
                x, target = x.to(device), target.to(device)
                val_pred = model(x)
                val_loss += loss_fn(val_pred, target, model).item()

        avg_val_loss = val_loss / len(valloader.dataset)
        plt_val_loss.append(avg_val_loss)

        # 若验证损失创新低则保存模型
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model, save_path)

        elapsed = time.time() - start_time
        print(f'[{epoch+1:03d}/{epochs:03d}] {elapsed:.2f}s  '
              f'TrainLoss: {plt_train_loss[-1]:.6f}  ValLoss: {plt_val_loss[-1]:.6f}')

    # 绘制损失曲线
    plt.plot(plt_train_loss, label='train')
    plt.plot(plt_val_loss, label='val')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    train_path = 'kc_house_data.csv'

    # 加载并打印数据集基本信息
    house_df = pd.read_csv(train_path)
    print(house_df.head())
    print(house_df.info())
    print(house_df.describe())

    # 构建完整数据集，再按 8:2 划分训练集与验证集
    house_dataset = HouseDataset(train_path)
    n_train = int(len(house_dataset) * 0.8)
    n_val = len(house_dataset) - n_train
    train_set, val_set = random_split(house_dataset, [n_train, n_val])

    config = {
        'n_epochs':   50,
        'batch_size': 25,
        'lr':         0.03,
        'save_path':  'model_save/model.pth',
    }

    trainloader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    valloader   = DataLoader(val_set,   batch_size=config['batch_size'], shuffle=False)

    # 特征维度 = 原始列数 - 3（删除 id、price、zipcode）
    in_dim = house_dataset[0][0].shape[0]
    model = MyNet(in_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=0.001)

    train_val(model, trainloader, valloader, optimizer,
              mse_loss, config['n_epochs'], device, config['save_path'])
