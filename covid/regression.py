# %%
# 标准库
import os
import csv
import time

# 第三方库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

# matplotlib 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ── 数据集 ────────────────────────────────────────────────────────────────────
# 继承 Dataset 必须实现三个方法：__init__ / __getitem__ / __len__
class CovidDataset(Dataset):

    def __init__(self, file_path, mode="train"):
        """
        mode: "train" | "val" | "test"
        划分策略：每 5 条数据中，第 0 条给 val，其余 4 条给 train（逢 5 取 1）
        """

        self.mode = mode

        # 读取 CSV，去掉表头（第 0 行）和 id 列（第 0 列），转为 float
        with open(file_path, "r") as f:
            ori_data = list(csv.reader(f))
            csv_data = np.array(ori_data[1:])[:, 1:].astype(float)

        # 按 mode 确定行索引，并提取标签列（最后一列）
        if self.mode == "train":
            indices = [i for i in range(len(csv_data)) if i % 5 != 0]
            self.y = torch.tensor(csv_data[indices, -1]).float()
            self.data = torch.tensor(csv_data[indices, :-1])#训练集不取最后一列
        elif self.mode == "val":
            indices = [i for i in range(len(csv_data)) if i % 5 == 0]
            self.y = torch.tensor(csv_data[indices, -1]).float()
            self.data = torch.tensor(csv_data[indices, :-1])  # 验证集也去掉最后一列标签
        else:  # test：无标签
            indices = list(range(len(csv_data)))
            self.data = torch.tensor(csv_data[indices])  # 测试集CSV本身没有标签列，不需要去掉

        # Z-score 标准化：(x - μ) / (σ + ε)，加 1e-8 防止除以 0
        self.data = (self.data - self.data.mean(dim=0, keepdim=True)) / \
                    (self.data.std(dim=0, keepdim=True) + 1e-8)

    def __getitem__(self, idx):
        # 训练/验证集返回 (特征, 标签)，测试集只返回特征
        if self.mode != "test":
            return self.data[idx].float(), self.y[idx]
        return self.data[idx].float()

    def __len__(self):
        return len(self.data)


# ── 模型 ────────────────────────────────────────────────────────────────────
# 继承 nn.Module 必须实现 __init__（定义层）和 forward（前向传播）
class CovidModel(nn.Module):

    def __init__(self, inDim):
        super(CovidModel, self).__init__()
        # 两层全连接：inDim → 64 → 1（回归输出单个值）
        self.fc1  = nn.Linear(inDim, 64)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        # 批量推理时输出 shape 为 (N, 1)，squeeze 成 (N,) 与标签对齐
        if x.dim() > 1:
            return x.squeeze(1)
        return x

#训练
def train_val(model, train_loader, val_loader,device, epochs,optimizer,loss,save_path):
    model = model.to(device)

    plt_train_loss = [] #记录每个epoch的训练损失
    plt_val_loss = [] #记录每个epoch的验证损失 

    min_val_loss = float("inf")

    for epoch in range(epochs): # 开始训练的号角
        train_loss = 0.0
        val_loss = 0.0
        start_time = time.time()

        model.train() #训练模型模式
        for batch_x, batch_y in train_loader:
            x, target = batch_x.to(device), batch_y.to(device)
            pred = model(x) #模型预测
            train_batch_loss = loss(pred, target) #计算损失
            train_batch_loss.backward() #反向传播
            optimizer.step() #更新参数
            optimizer.zero_grad() #清空梯度 如果不清空就会梯度累加
            train_loss += train_batch_loss.cpu().item() #把损失从GPU上拿下来，转成python的float类型

        plt_train_loss.append(train_loss/len(train_loader)) #记录训练损失

#验证,注意验证集不更新模型
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                x, target = batch_x.to(device), batch_y.to(device)
                pred = model(x)
                val_batch_loss = loss(pred, target)
                val_loss += val_batch_loss.cpu().item()

        plt_val_loss.append(val_loss / len(val_loader)) #记录验证损失
        
        #保存模型
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path) #保存模型参数

        print("[%0.3d/%0.3d] %2.2f sec(s) Trainloss: %.6f |Valloss: %.6f" % \
              (epoch, epochs, time.time() - start_time,plt_train_loss[-1],plt_val_loss[-1]))

    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title("loss图")
    plt.legend(["train","val"])
    plt.show()

def evaluate(save_path, test_loader, device, rel_path, inDim): # 得出测试结果文件
    model = CovidModel(inDim=inDim)
    model.load_state_dict(torch.load(save_path))
    model = model.to(device)
    rel = []
    with torch.no_grad(): #不计算梯度
        for x in test_loader:
            pred = model(x.to(device))
            rel.append(pred.cpu().item())#把预测值从GPU上拿下来，转成python的float类型
    print(rel)
    with open(rel_path,'w', newline = '') as f: # 写入文件时，newline = '' 避免在 Windows 上添加额外的空行
        csvWriter = csv.writer(f)
        csvWriter.writerow(["id","tested_positive"])
        for i , value in enumerate(rel): #这个写法可以同时取到下标和值
            csvWriter.writerow([str(i),str(value)])
    print("文件已保存到{}".format(rel_path)) #print("文件已保存到{}" + rel_path) 字符串可以直接相加

# ── 主程序 ───────────────────────────────────────────────────────────────────

# ① 文件路径：兼容脚本运行（__file__ 存在）和 Jupyter/交互式运行（__file__ 不存在）
try:
    _dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _dir = os.path.abspath("")

train_file = os.path.join(_dir, "covid.train.csv")
test_file  = os.path.join(_dir, "covid.test.csv")

# ② 构建数据集：train/val 共用同一份训练 CSV，按索引划分；test 无标签
train_dataset = CovidDataset(train_file, mode="train")
val_dataset   = CovidDataset(train_file, mode="val")
test_dataset  = CovidDataset(test_file,  mode="test")

# ③ 构建 DataLoader：训练集打乱顺序以增强随机性，验证集保持原序
batchsize    = 16
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batchsize, shuffle=False)
test_loader = DataLoader(test_dataset,batch_size = 1, shuffle = False) #注意这里shuffle一定要改为false 同时batchsize = 1  因为训练集的x与y对应的 所以可以打乱  但是测试集不能打乱不然有可能会出现预测的第10个值放在第一个格子里

# ④ 设备选择：优先使用 GPU 加速，无 GPU 时退回 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# ⑤ 超参数集中管理，方便后续调参
config = {
    "lr":        0.001,               # 学习率
    "epochs":    20,                  # 训练轮数
    "monmentum": 0.9,                 # SGD 动量，加速收敛并抑制震荡
    "save_path": "model_save/best_model.pth",
    "rel_path" : "model_save/pred.csv"
}

# ⑥ 实例化模型、损失函数、优化器
# inDim 自动从数据集特征维度读取，避免硬编码
model     = CovidModel(inDim=train_dataset.data.shape[1]).to(device)
loss      = nn.MSELoss()              # 均方误差，回归任务的标准损失函数
optimizer = optim.SGD(               # 随机梯度下降
    model.parameters(),
    lr=config["lr"],
    momentum=config["monmentum"]
)

# ⑦ 开始训练；训练结束后自动绘制 loss 曲线并保存最优模型
train_val(model, train_loader, val_loader, device,
          config["epochs"], optimizer, loss, config["save_path"])

#验证过程 
evaluate(config["save_path"], test_loader, device, config["rel_path"], train_dataset.data.shape[1])#用最好模型来进行测试











