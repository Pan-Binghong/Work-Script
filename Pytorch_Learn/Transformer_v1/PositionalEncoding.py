#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Positional_Encoding.py
@Time    :   2024/09/26 11:23:36
@Author  :   pan binghong 
@Email   :   19909442097@163.com
@description   :   Transformer中的绝对位置编码底层代码实现
'''
import torch
import torch.nn as nn
import math
from transformers import BertTokenizer
import os

file_path = os.path.abspath(__file__)

dir = os.path.dirname(file_path)

os.chdir(dir)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        ''''
        :param d_model: 模型的维度
        :param max_len: 序列的最大长度
        '''
        super(PositionalEncoding, self).__init__()

        # 创建一个形状为 (max_len, d_model) 的矩阵, 用于存储位置编码
        pe = torch.zeros(max_len, d_model)

        # 创建一个形状为 (max_len, 1) 的矩阵, 用于存储位置信息, 保存索引值 e.g.[0, 1, 2, ... , max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 这段代码计算位置编码中的两个频率。具体来说，它生成一个从0到d_model（不包括d_model）的偶数序列，
        # 然后将这些偶数转换为浮点数，并乘以一个常数因子 (-math.log(10000.0) / d_model)。
        # 这个常数因子是通过对10000取自然对数并除以d_model得到的。
        # 最后，通过torch.exp函数计算这些值的指数，得到最终的div_term。
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 应用正弦函数 得到偶数索引位置编码
        pe[:, 0::2] = torch.sin(position * div_term)

        # 应用余弦函数 得到奇数索引位置编码
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加一个 batch 维度, 使其能够与输入一起使用
        pe = pe.unsqueeze(0)

        # 将位置编码矩阵注册为一个参数, 并将其添加到模型参数列表中
        self.register_buffer('pe', pe)

    def forward(self, x):
        ''''
        :param x: 输入的序列张量, shape为: <batch_size, seq_len, d_model>
        :return: 输出的序列张量, shape为: <batch_size, seq_len, d_model>
        '''
        x = x + self.pe[:, :x.size(1), :]
        return x
    
if __name__ == '__main__':
    # 初始化参数
    d_model = 512
    max_len = 2048

    # 初始化位置编码
    pos_encoder = PositionalEncoding(d_model, max_len)

    # 初始化 tokenizer (这里以 Bert 为例)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./cache')

    # 使用输入"hello world"
    input_text = "hello world"
    input_ids = torch.tensor([tokenizer.encode(input_text, add_special_tokens=True)])

    # 创建一个形状为 (1, seq_len, d_model) 的零矩阵
    x = torch.zeros(1, input_ids.size(1), d_model)

    # 应用位置编码
    output = pos_encoder(x)

    print("Input Text:", input_text)
    print("Input IDs Shape:", input_ids.shape)
    print("Output Shape:", output.shape)
    print("Output:", output)