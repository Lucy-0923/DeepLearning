import random #随机数
import torch #深度学习框架

import torch.nn as nn
import numpy as np #数值计算库
import os #操作系统库
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.fx  # 提前导入
from timm.models.vision_transformer import PatchEmbed, Block

from PIL import Image #读取图片
from torch.utils.data import Dataset, DataLoader #数据集和数据加载器
from tqdm import tqdm #进度条 查看读取数据的进度条 显示循环进度,避免盲等
from torchvision import transforms #数据增强
from model_utils.model import initialize_model

import matplotlib.pyplot as plt
import time

# matplotlib 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
#################################################################
seed_everything(0)
###############################################

HW = 224

# def read_file(path):#读取文件  大循环
#     for i in tqdm(range(11)): #这一步是需要看文件夹得出的11
#         file_dir = path + "/%02d" % i #列出文件夹下所有文件名字
#         file_list = os.listdir(file_dir)


#         xi = np.zeros((len(file_list), HW, HW, 3), dtype = np.uint8) #3表示3个通道的RGB图片 ,dtype表示数据类型是无符号8位整数
#         yi = np.zeros((len(file_list)), dtype = np.uint8) #11表示11个类别 ,dtype表示数据类型是无符号8位整数    因为图片的三原色都是整数


#         #列出文件夹下所有文件名字
#         for j, img_name in enumerate(tqdm(file_list)): #小循环
#             img_path = os.path.join(file_dir, img_name) 
#             img = Image.open(img_path) #此时图片的大小是512*512 但是我们需要的是224*224
#             img = img.resize((HW, HW))#将图片的大小调整为224*224
#             xi[j, ...] = img
#             yi[j] = i

#         if i == 0:
#             X = xi
#             Y = yi
#         else:
#             X = np.concatenate((X, xi), axis=0)  #将X和xi按行拼接   维度知识竖的方向是0维 横的方向是1维
#             Y = np.concatenate((Y, yi), axis=0)  #将Y和yi按行拼接
#     print("读到了%d个数据"%len(Y))
#     return X, Y

train_transform = transforms.Compose( #训练的时候需要进行数据增广
    [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),#随机裁剪224*224的图片
        transforms.RandomRotation(50), #随机旋转50度
        transforms.ToTensor(), #将图片转化为为张量
    ]
)#compose表示将多个变换组合起来

val_transform = transforms.Compose( #验证和测试的时候需要用原图 训练的时候可以进行变换
    [
        transforms.ToPILImage(),
        transforms.ToTensor(), #将图片转化为为张量
    ]
)#compose表示将多个变换组合起来

class food_Dataset(Dataset):
    def __init__(self, path, mode = "train"): #初始化数据集
        self.mode == mode
        if mode == "semi":
            self.X = self.read_file(path)
        else:
            self.X, self.Y = self.read_file(path)
            self.Y = torch.LongTensor(self.Y) #将标签转化为长整型
        if mode == "train":
            self.transform = train_transform
        else:
            self.transform = val_transform

   
    
#直接把上面的read_file函数复制过来,修改一下,就变成了半监督学习的read_file函数
    def read_file(self,path):#读取文件  大循环

        if self.mode == "semi":
                file_list = os.listdir(path)
                xi = np.zeros((len(file_list), HW, HW, 3), dtype = np.uint8) #3表示3个通道的RGB图片 ,dtype表示数据类型是无符号8位整数
                #列出文件夹下所有文件名字
                for j, img_name in enumerate(tqdm(file_list)): #小循环
                    img_path = os.path.join(path, img_name) 
                    img = Image.open(img_path) #此时图片的大小是512*512 但是我们需要的是224*224
                    img = img.resize((HW, HW))#将图片的大小调整为224*224
                    xi[j, ...] = img
                print("读到了%d个数据"%len(Y))
                return X, Y
        else:
            for i in tqdm(range(11)): #这一步是需要看文件夹得出的11
                file_list = os.listdir(path)
                xi = np.zeros((len(file_list), HW, HW, 3), dtype = np.uint8) #3表示3个通道的RGB图片 ,dtype表示数据类型是无符号8位整数
                yi = np.zeros(len(file_list), dtype=np.uint8)              
                #列出文件夹下所有文件名字
                for j, img_name in enumerate(tqdm(file_list)): #小循环
                    img_path = os.path.join(path, img_name) 
                    img = Image.open(img_path) #此时图片的大小是512*512 但是我们需要的是224*224
                    img = img.resize((HW, HW))#将图片的大小调整为224*224
                    xi[j, ...] = img
                    yi[j] = i

                if i == 0:
                    X = xi
                    Y = yi
                else:
                    X = np.concatenate((X, xi), axis=0)  #将X和xi按行拼接   维度知识竖的方向是0维 横的方向是1维
                    Y = np.concatenate((Y, yi), axis=0)  #将Y和yi按行拼接
            print("读到了%d个数据"%len(Y))
            return X, Y

    def __getitem__(self, item): #获取数据集中的一个样本
        if self.mode == "semi":
            return self.transform(self.X[item]),self.X[item]
        else:
            return self.transform(self.X[item]), self.Y[item]


    def __len__(self): #获取数据集的样本数量
      return  len(self.X)



class semiDataset(Dataset):
    def __init__(self, no_label_loder, model, device,thres = 0.1):
        x, y = self.getlabel(no_label_loder, model, device,thres)
        if x == []:
            self.flag = False
        else:
            self.flag = True
            self.X = np.array(x)
            self.Y = torch.LongTensor(y)
            self.transform = train_transform


    def get_label(self,no_label_loder,model,device,thres = 0.1):
        model = model.to(device)
        pred_prob = []
        labels = []
        x = []
        y = []
        soft = nn.Softmax()
        with torch.no_grad():#一旦让一个数据应该模型都会积攒梯度,如果对模型训练没有任何帮助就不让他产生梯度
            for bat_x, _ in no_label_loder:
                bat_x = bat_x.to(device)
                pred = model(bat_x)
                pred_soft = soft(pred)
                pred_max ,pred_value = pred_soft.max(1)
                pred_prob.extend(pred_max.cpu().numpy.tolist())
                labels.extend(pred_value.cpu().numpy.tolist())
        
        for index, prob in enumerate(pred_prob):
            if prob > thres:
                x.append(no_label_loder.dataset[index][1])
                y.append(labels[index])
        return x, y
    
    def __getitem__(self, item): #获取数据集中的一个样本
        return self.transform(self.X[item]),self.X[item]

    def __len__(self): #获取数据集的样本数量
      return  len(self.X)



def get_semi_loader(no_label_loder, model, device,thres = 0.1):
    semiset = semiDataset(no_label_loder, model, device,thres)
    if semiset.flag == False:
        return None
    else:
        semi_loader = DataLoader(semiset, batch_size=16, shuffle=False)
        return semi_loader
        

class myModel(nn.Module):
    def __init__(self, num_class):
        super(myModel, self).__init__()
        #3 * 224 * 224 -> 512 * 7 * 7 -> 拉直 -> 全连接 
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)#卷积
        self.bn1 = nn.BatchNorm2d(64)#归一化
        self.relu = nn.ReLU()#激活函数
        self.pool1 = nn.MaxPool2d(2, 2)#池化层 64 * 112 * 112

        # self.conv2 = nn.Conv2d(64, 28, 3, 1, 1)#卷积
        # self.bn2 = nn.BatchNorm2d(864#归一化
        # self.relu2 = nn.ReLU()#激活函数
        # self.pool = nn.MaxPool2d(2, 2)#池化层 128*56*56

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)#池化层 128*56*56
        )#合起来写

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)#池化层 256*28*28
        )#合起来写

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)#池化层 512*14*14
        )#合起来写

        self.pool2 = nn.MaxPool2d(2)#池化层 512*7*7
        self.fc1 = nn.Linear(512*7*7, 1000)#全连接层 25088  -> 1000
        self.relu1 = nn.ReLU()#激活函数
        self.fc2 = nn.Linear(1000, num_class)#全连接层 1000  -> 11

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool2(x)
        x = x.view(x.size()[0], -1) #将x拉直    
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

def train_val(model, train_loader, val_loader, no_label_loader,device, epochs, optimizer, loss, thres,save_path):
    model = model.to(device)
    semi_loader = []
    plt_train_loss = [] #记录每个epoch的训练损失
    plt_val_loss = [] #记录每个epoch的验证损失 

    plt_train_acc = [] #记录每个epoch的训练准确率
    plt_val_acc = [] #记录每个epoch的验证准确率 


    min_val_loss = float("inf")

    for epoch in range(epochs): # 开始训练的号角
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        semi_loss = 0.0
        semi_acc = 0.0

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
            train_acc += np.sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy())

        plt_train_loss.append(train_loss / train_loader.__len__())
        plt_train_acc.append(train_acc/train_loader.dataset.__len__()) 


        if semi_loader!=None:
            for batch_x, batch_y in semi_loader:
                x, target = batch_x.to(device), batch_y.to(device)
                pred = model(x) #模型预测
                semi_batch_loss = loss(pred, target) #计算损失
                semi_batch_loss.backward() #反向传播
                optimizer.step() #更新参数
                optimizer.zero_grad() #清空梯度 如果不清空就会梯度累加
                semi_loss += train_batch_loss.cpu().item() #把损失从GPU上拿下来，转成python的float类型
                semi_acc += np.sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy())
            print("半监督数据集的训练准确率为",semi_acc/train_loader.dataset.__init__())

#验证,注意验证集不更新模型
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                x, target = batch_x.to(device), batch_y.to(device)
                pred = model(x)
                val_batch_loss = loss(pred, target)
                val_loss += val_batch_loss.cpu().item()
                val_acc += np.sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy())

        plt_val_loss.append(val_loss / val_loader.dataset.__len__())
        plt_val_acc.append(val_acc / val_loader.dataset.__len__())
        
        if epoch%5 == 0 and plt_val_acc[-1] >0.1: #实际应用的时候至少大于0.6
            semi_loader = get_semi_loader(no_label_loader,model, device, thres = 0.1)

         #在实际中并不是每一轮都进行半监督，不然会很慢

        #保存模型，这里用平均val_loss比较
        avg_val_loss = val_loss / len(val_loader.dataset)
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path) #保存模型参数

        print("[%0.3d/%0.3d] %2.2f sec(s) Trainloss: %.6f |Valloss: %.6f |Trainacc: %.6f |Valacc: %.6f" % \
              (epoch, epochs, time.time() - start_time,plt_train_loss[-1],plt_val_loss[-1],plt_train_acc[-1],plt_val_acc[-1]))

    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title("loss图")
    plt.legend(["train","val"])
    plt.show()

    plt.plot(plt_train_acc)
    plt.plot(plt_val_acc)
    plt.title("acc图")
    plt.legend(["train","val"])
    plt.show()

# path = r"D:\DeepLearning\第三节，回归实战代码\第四五节_分类代码\food_classification\food-11\training\labeled"#r表示去除转移字符 
train_path = r"D:\DeepLearning\第三节，回归实战代码\第四五节_分类代码\food_classification\food-11\training\labeled"#r表示去除转移字符 path = r"D:\DeepLearning\第三节，回归实战代码\第四五节_分类代码\food_classification\food-11\training\labeled"#r表示去除转移字符 
val_path = r"D:\DeepLearning\第三节，回归实战代码\第四五节_分类代码\food_classification\food-11\validation"#r表示去除转移字符 path = r"D:\DeepLearning\第三节，回归实战代码\第四五节_分类代码\food_classification\food-11\training\labeled"#r表示去除转移字符 
np_label_path = r"D:\DeepLearning\第三节，回归实战代码\第四五节_分类代码\food_classification\food-11\training\unlabeled\00" #r表示去除转移字符 


train_set = food_Dataset(train_path,"train")
val_set = food_Dataset(val_path,"val")
no_label_set = food_Dataset(np_label_path,"semi")

train_loader = DataLoader(train_set, batch_size = 16, shuffle=True) #将数据集转换为数据加载器 每个批次16个样本 随机打乱,变为一批数据
val_loader = DataLoader(val_set, batch_size = 16, shuffle=True) #将数据集转换为数据加载器 每个批次16个样本 随机打乱,变为一批数据
no_label_loader = DataLoader(no_label_set, batch_size = 16, shuffle=False) #这里一定要等于False因为是对于无标签数据集中每个标签进行预测然后打上标签，如果变为混乱那么打上的标签可能不对于了

# model = myModel(11)
model, _ = initialize_model("resnet18", 11, use_pretrained=True)
#, - 用来承接第二个返回值

lr = 0.001
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-4)  #优化器,加上权重衰减
device = "cuda" if torch.cuda.is_available() else "cpu" #判断是否有cuda
print(f"当前使用设备: {device}")
if device == "cuda":
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
save_path = "model_save/best_model.pth"
epochs = 15
thres = 0.1 #实际应用的时候 thres一定要打上0.99这么高，因为你打上的标签一定要准
semi_set = semiDataset(no_label_loader,model, device, thres = 0.1)


train_val(model, train_loader, val_loader, no_label_loader,device, epochs, optimizer, loss, thres,save_path)



