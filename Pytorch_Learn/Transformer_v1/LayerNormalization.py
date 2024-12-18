#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   LayerNormalization.py
@Time    :   2024/09/27 15:17:48
@Author  :   pan binghong 
@Email   :   19909442097@163.com
@description   :   
'''
import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self, features, eps=1e-6):
        '''
        初始化层归一化模块
        :param features: 特征维度大小
        :param eps: 防止除零的小常数
        '''
        super(LayerNormalization, self).__init__()  # 调用父类nn.Module的初始化方法
        self.eps = eps  # 设置防止除零的小常数
        self.gain = nn.Parameter(torch.ones(features))  # 初始化增益参数，形状为(features,)，初始值为1
        self.bias = nn.Parameter(torch.zeros(features))  # 初始化偏置参数，形状为(features,)，初始值为0

    def forward(self, x):
        '''
        前向传播函数
        :param x: 输入张量，形状为(batch_size, seq_len, features)
        :return: 归一化后的输出张量
        '''
        mean = x.mean(dim=-1, keepdim=True)  # 计算输入张量在最后一个维度上的均值，保持维度
        std = x.std(dim=-1, keepdim=True)    # 计算输入张量在最后一个维度上的标准差，保持维度
        # print(f'layer normalization \ngain shape: \n{self.gain.shape}, \nbias shape: {self.bias.shape}, \ninput shape: {x.shape}')
        return self.gain * (x - mean) /(std +self.eps) + self.bias  # 归一化公式

if __name__ == "__main__":
    batch_size = 32
    seq_len = 2048
    features = 4096

    # 创建一个简单的输入张量
    x = torch.randn(batch_size, seq_len, features)  # 随机初始化输入张量
    # 初始化层归一化层
    ln = LayerNormalization(features)

    # 应用层归一化
    normalized_x = ln(x)

    # 打印原始和归一化后的张量
    print("原始输入张量:")
    print(x)
    print("\n归一化后的输出张量:")
    print(normalized_x)
    print("\n归一化后的维度:")
    print(normalized_x.shape)