#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Encoder.py
@Time    :   2024/09/29 08:39:15
@Author  :   pan binghong 
@Email   :   19909442097@163.com
@description   :   
'''

import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from FeedForwardNetwork import FeedForwardNetwork
from LayerNormalization import LayerNormalization #在残差连接模块中完成
from ResidualConnection import ResidualConnection

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_size, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.pos_ffn = FeedForwardNetwork(d_model, hidden_size, dropout)
        self.residual = nn.ModuleList([
            ResidualConnection(d_model, dropout),
            ResidualConnection(d_model, dropout)
        ])
    
    def forward(self, x, mask=None):
        x = self.residual[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.residual[1](x, self.pos_ffn)
        return x

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(Encoder, self).__init__()
        self.encoder_layer = nn.ModuleList([
            encoder_layer for _ in range(num_layers)
        ])
        self.norm = LayerNormalization(encoder_layer.d_model)

    def forward(self, src, mask=None):
        for layer in self.encoder_layer:
            src = layer(src, mask)
        return self.norm(src)
    

if __name__ == '__main__':
    x = torch.rand((1, 10, 512))
    
    # 定义模型参数
    d_model = 512  # 嵌入维度
    num_heads = 8  # 注意力头数
    hidden_size = 2048  # 前馈网络的隐藏层大小
    num_layers = 6  # 编码器层数
    
    # 创建编码器层和编码器
    encoder_layer = EncoderLayer(d_model, num_heads, hidden_size)
    encoder = Encoder(encoder_layer, num_layers)
    
    # 打印编码器结构
    print("编码器结构：")
    print(encoder)
    
    # 打印编码器层结构
    print("\n编码器层结构：")
    print(encoder_layer)
    
    # 前向传播
    output = encoder(x)
    print(output.shape)
