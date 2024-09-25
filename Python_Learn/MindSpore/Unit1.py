"""
@Description :   
@Author      :   Pan BingHong
@Time        :   2024/09/25 21:58:00
"""

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.mint as mint

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 第一层卷积，输入通道1，输出通道6，卷积核大小5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # 第二层卷积，输入通道6，输出通道16，卷积核大小5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 第三层全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 输入特征数为16*5*5
        self.fc2 = nn.Linear(120, 84)           # 第二层全连接
        self.fc3 = nn.Linear(84, 10)             # 输出层，10个类别

    def forward(self, x):
        # 第一个卷积层 + 激活 + 池化
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # 第二个卷积层 + 激活 + 池化
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # 展平
        x = x.view(x.size(0), -1)  # 将多维输入一维化
        
        # 全连接层 + 激活
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # 输出层
        x = self.fc3(x)
        return x

# 实例化模型
model = LeNet5()
print(model)
