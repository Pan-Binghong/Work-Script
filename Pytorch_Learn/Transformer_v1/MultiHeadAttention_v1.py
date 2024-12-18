#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   MultiHeadAttention.py
@Time    :   2024/09/27 09:50:43
@Author  :   pan binghong 
@Email   :   19909442097@163.com
@description   :   
'''
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer

# 获取当前文件的绝对路径
file_path = os.path.abspath(__file__)

# 获取当前文件所在的目录路径
dir = os.path.dirname(file_path)

# 将当前工作目录更改为当前文件所在的目录
os.chdir(dir)

import torch
import torch.nn as nn
from transformers import BertTokenizer

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        '''
        初始化多头注意力机制
        :param embed_size: 输入嵌入向量的维度
        :param heads: 多头注意力机制的头数
        '''
        super(MultiHeadAttention, self).__init__()

        assert embed_size % heads == 0, "嵌入维度必须能被头数整除"

        self.embed_size = embed_size
        self.heads = heads
        # 每个头的维度
        self.heads_dim = embed_size // heads

        self.value = nn.Linear(embed_size, embed_size, bias=False)
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.query = nn.Linear(embed_size, embed_size, bias=False)

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, query, keys, values, mask):
        N = query.shape[0]  # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # 初始化Q, K, V矩阵
        values = self.value(values)  # [N, value_len, embed_size]
        keys = self.key(keys)        # [N, key_len, embed_size]
        query = self.query(query)    # [N, query_len, embed_size]
        
        # 将 Q, K, V 的形状恢复为 (N, heads, seq_len, heads_dim)
        values = values.view(N, value_len, self.heads, self.heads_dim).transpose(1, 2)  # [N, heads, value_len, heads_dim]
        keys = keys.view(N, key_len, self.heads, self.heads_dim).transpose(1, 2)        # [N, heads, key_len, heads_dim]
        query = query.view(N, query_len, self.heads, self.heads_dim).transpose(1, 2)    # [N, heads, query_len, heads_dim]

        # 计算注意力分数
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys])  # [N, heads, query_len, key_len]

        # 应用注意力机制
        print(mask)
        print(energy.shape)
        # 应用注意力机制
        if mask is not None:
            # Ensure mask has the correct shape
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1).expand(-1, self.heads, query_len, key_len)  # [N, heads, query_len, key_len]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1).expand(-1, self.heads, query_len, key_len)  # [N, heads, query_len, key_len]
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)  # [N, heads, query_len, key_len]

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])  # [N, heads, query_len, heads_dim]
        out = out.transpose(1, 2).reshape(N, query_len, self.heads * self.heads_dim)  # [N, query_len, embed_size]

        out = self.fc_out(out)  # [N, query_len, embed_size]
        return out
    
    
if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./cache')

    input_text = "Hello world!"

    # 初始化随机keys, values, query
    tokens = tokenizer.tokenize(input_text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = torch.tensor([token_ids]).long()

    batch_size, seq_length = token_ids.shape
    embed_size = 512
    heads = 8

    values = torch.rand(batch_size, seq_length, embed_size)
    keys = torch.rand(batch_size, seq_length, embed_size)
    query = torch.rand(batch_size, seq_length, embed_size)
    mask = None

    attention = MultiHeadAttention(embed_size, heads)
    out = attention(query, keys, values, mask)

    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(out.shape)
    print(f"Multi-head Attention Output: \n{out}")





