#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ResidualConnection.py
@Time    :   2024/09/27 15:33:49
@Author  :   pan binghong 
@Email   :   19909442097@163.com
@description   :   
'''
import torch
import torch.nn as nn
from LayerNormalization import LayerNormalization


class ResidualConnection(nn.Module):
    def __init__(self, size, dropout):
        super(ResidualConnection, self).__init__()  # 正确调用父类构造函数
        self.norm = LayerNormalization(size)  # 在父类构造函数之后设置属性
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

if __name__ == '__main__':
    
    size = 512
    dropout = 0.1
    
    residual_module = ResidualConnection(size, dropout)

    x = torch.rand(32, 10, size)
    sublayer = nn.Linear(size, size)

    output = residual_module(x, sublayer)
    print(f'output shape: \n{output.shape}')
    print(f'out: \n{output}')

