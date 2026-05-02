# COVID-19 回归实战代码调试笔记

## 错误汇总与解决方案

### 错误1：evaluate() 函数参数缺失

**错误信息：**
```
TypeError: evaluate() missing 3 required positional arguments: 'test_loader', 'device', and 'rel_path'
```

**问题分析：**
`evaluate()` 函数定义了4个参数，但调用时只传入了1个参数。

**解决方案：**
```python
# 修改前
evaluate(config["save_path"],)

# 修改后
evaluate(config["save_path"], test_loader, device, config["rel_path"])
```

---

### 错误2：模型加载方式错误

**错误信息：**
```
AttributeError: 'collections.OrderedDict' object has no attribute 'to'
```

**问题分析：**
代码使用 `torch.save(model.state_dict(), save_path)` 保存的是模型的参数字典（state_dict），但加载时却直接当作模型对象处理。

**解决方案：**
```python
# 修改前
model = torch.load(save_path).to(device)

# 修改后
model = CovidModel(inDim=...)
model.load_state_dict(torch.load(save_path))
model = model.to(device)
```

---

### 错误3：训练集与测试集维度不匹配

**错误信息：**
```
RuntimeError: size mismatch for fc1.weight: copying a param with shape torch.Size([64, 94]) from checkpoint, the shape in current model is torch.Size([64, 93])
```

**问题分析：**
测试集没有去掉标签列，导致特征维度比训练集多1。

**解决方案：**
```python
# 在 CovidDataset 类中，测试集也需要去掉最后一列标签
self.data = torch.tensor(csv_data[indices, :-1])
```

---

### 错误4：验证集缺少 data 属性

**错误信息：**
```
AttributeError: 'CovidDataset' object has no attribute 'data'
```

**问题分析：**
在 `CovidDataset` 类的 `__init__` 方法中，`val` 模式只设置了 `self.y`，没有设置 `self.data`。

**解决方案：**
```python
elif self.mode == "val":
    indices = [i for i in range(len(csv_data)) if i % 5 == 0]
    self.y = torch.tensor(csv_data[indices, -1]).float()
    self.data = torch.tensor(csv_data[indices, :-1])  # 添加这行
```

---

### 错误5：evaluate 函数维度不匹配

**错误信息：**
```
RuntimeError: size mismatch for fc1.weight: copying a param with shape torch.Size([64, 93]) from checkpoint, the shape in current model is torch.Size([64, 92])
```

**问题分析：**
在 `evaluate()` 函数中使用测试集的维度创建模型，但测试集和训练集的CSV结构可能不同（测试集可能没有标签列）。

**解决方案：**
```python
# 修改 evaluate 函数，接收输入维度作为参数
def evaluate(save_path, test_loader, device, rel_path, inDim):
    model = CovidModel(inDim=inDim)
    model.load_state_dict(torch.load(save_path))
    model = model.to(device)

# 调用时传入训练集的维度
evaluate(config["save_path"], test_loader, device, config["rel_path"], train_dataset.data.shape[1])
```

---

### 错误6：矩阵乘法维度不匹配

**错误信息：**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x92 and 93x64)
```

**问题分析：**
测试集CSV文件本身就没有标签列，但代码仍然执行了 `[:, :-1]`，导致多去掉了一列。

**解决方案：**
```python
# 修改前（错误）
self.data = torch.tensor(csv_data[indices, :-1])

# 修改后（正确）
self.data = torch.tensor(csv_data[indices])  # 测试集CSV本身没有标签列
```

---

## 关键知识点总结

### 1. PyTorch 模型保存与加载

**保存模型参数：**
```python
torch.save(model.state_dict(), path)  # 保存的是 OrderedDict
```

**加载模型参数：**
```python
model = ModelClass(inDim=...)           # 先创建模型实例
model.load_state_dict(torch.load(path)) # 再加载参数
model = model.to(device)                # 最后移动到设备
```

### 2. 数据集处理注意事项

- 训练集和验证集：需要分离特征和标签（去掉最后一列）
- 测试集：CSV文件本身可能没有标签列，直接使用全部列

### 3. 维度一致性

确保训练、验证、测试三个阶段使用相同的特征维度：
- 模型输入维度应与训练集特征维度一致
- 测试集特征维度应与训练集特征维度一致

---

## 最终代码修改位置

| 修改位置 | 问题描述 | 修改内容 |
| :--- | :--- | :--- |
| `CovidDataset.__init__` val模式 | 缺少 `self.data` | 添加 `self.data = torch.tensor(csv_data[indices, :-1])` |
| `CovidDataset.__init__` test模式 | 错误地去掉最后一列 | 修改为 `self.data = torch.tensor(csv_data[indices])` |
| `evaluate()` 函数 | 直接加载state_dict | 修改为先创建模型再加载参数 |
| `evaluate()` 函数 | 维度不匹配 | 添加 `inDim` 参数 |
| 主程序调用 | 参数缺失 | 添加完整参数列表 |