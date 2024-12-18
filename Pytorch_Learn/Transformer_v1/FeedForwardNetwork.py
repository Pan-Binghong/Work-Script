#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   FeedForwardNetwork.py
@Time    :   2024/09/27 14:46:48
@Author  :   pan binghong 
@Email   :   19909442097@163.com
@description   :   
'''
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer

file_path = os.path.abspath(__file__)
dir = os.path.dirname(file_path)
os.chdir(dir)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, hidden_size, dropout=0.1):
        '''
        :param d_model:输入的特征维度大小
        :param hidden_size:隐藏层大小
        :param dropout:dropout概率
        '''
        super(FeedForwardNetwork, self).__init__()

        self.liner1 = nn.Linear(d_model, hidden_size)
        self.relu = nn.ReLU()
        self.liner2 = nn.Linear(hidden_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        :param x:输入的特征
        :return:输出的特征
        '''
        x = self.liner1(x)
        x = self.relu(x)
        x = self.liner2(x)
        x = self.dropout(x)
        return x

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',cache_dir='./cache')
    input_text = "Hello world"
    tokens = tokenizer.tokenize(input_text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = torch.tensor([token_ids]).long()

    d_model = token_ids.shape[1]
    hidden_size = 256

    ff_netword = FeedForwardNetwork(d_model, hidden_size)

    output = ff_netword(token_ids.float())
    print(f'Input tokens: \n{tokens}')
    print(f'Input token_ids: \n{token_ids}')
    print(f'Output from FeedForwardNetwork: \n{output}')
    print(f'Output shape: \n{output.shape}')
