from torchvision import models
import torch
import torch.nn as nn

class myModel(nn.Module):
    def __init__(self, num_cls):    # num_cls 是分类数
        super(myModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4,padding=2)
        self.pool1 = nn.MaxPool2d(3, stride=2)

        self.conv2 = nn.Conv2d(64,192,5,1,2)
        self.pool2 = nn.MaxPool2d(3, stride=2)

        self.conv3 = nn.Conv2d(192,384,3,1,1)
        self.conv4 = nn.Conv2d(384,256,3,1,1)
        self.conv5 = nn.Conv2d(256, 256, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(3, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d(output_size=6)
        
        self.adapool = nn.AdaptiveAvgPool2d(output_size=6)
        self.fc1 = nn.Linear(9216,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,num_cls)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.pool4(x)  #batch*256*6*6

        x = x.view(x.size()[0], -1) # 展开为一维向量 ,x.size()[0]是batch_size,保持batch不变,  -1:全部放在第二维上
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = myModel(num_cls=1000)
data = torch.ones((4, 3, 224, 224))
pred = model(data)

def get_parameter_number(model): #算参数量
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

print(get_parameter_number(model.pool1))
#计算输出维度 计算参数量
