#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   MultiHeadAttention.py
@Time    :   2024/09/27 09:50:43
@Author  :   pan binghong 
@Email   :   19909442097@163.com
@description   :   
'''
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.heads_dim = embed_size // heads

        assert (self.heads_dim * heads == embed_size), "Embed size needs to be divisible by heads"

        self.value = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.query = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, query, keys, values, mask):
        N = query.shape[0]  # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Initialize Q, K, V matrices
        values = self.value(values)  # [N, value_len, embed_size]
        keys = self.key(keys)        # [N, key_len, embed_size]
        query = self.query(query)    # [N, query_len, embed_size]
        
        # Reshape Q, K, V to (N, heads, seq_len, heads_dim)
        values = values.view(N, value_len, self.heads, self.heads_dim).transpose(1, 2)  # [N, heads, value_len, heads_dim]
        keys = keys.view(N, key_len, self.heads, self.heads_dim).transpose(1, 2)        # [N, heads, key_len, heads_dim]
        query = query.view(N, query_len, self.heads, self.heads_dim).transpose(1, 2)    # [N, heads, query_len, heads_dim]

        # Calculate attention scores
        energy = torch.matmul(query, keys.transpose(2, 3))  # [N, heads, query_len, key_len]

        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # Adjust mask for attention dimensions
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        attention = torch.softmax(energy / (self.heads_dim ** (1/2)), dim=3)  # [N, heads, query_len, key_len]

        out = torch.matmul(attention, values)  # [N, heads, query_len, heads_dim]
        out = out.transpose(1, 2).reshape(N, query_len, self.heads * self.heads_dim)  # [N, query_len, embed_size]

        out = self.fc_out(out)  # [N, query_len, embed_size]
        return out

if __name__ == '__main__':
    # Test MultiHeadAttention module
    embed_size = 512
    heads = 8
    batch_size = 32
    query_len = 10  # Length of query
    key_len = 12    # Length of keys and values

    # Generate random input data
    query = torch.randn((batch_size, query_len, embed_size))
    keys = torch.randn((batch_size, key_len, embed_size))
    values = torch.randn((batch_size, key_len, embed_size))

    # Example mask: a binary mask of shape (batch_size, 1, 1, key_len) that applies across all heads and query positions
    mask = torch.ones((batch_size, key_len))
    mask[:, 5:] = 0  # Mask out the last part of the sequence

    # Adjust mask shape for multi-head attention
    mask = mask.unsqueeze(1).unsqueeze(1)  # Shape: (batch_size, 1, 1, key_len)

    # Initialize MultiHeadAttention module
    mha = MultiHeadAttention(embed_size, heads)

    # Forward pass
    output = mha(query, keys, values, mask=mask)

    # Print output shape
    print(f"Output shape: {output.shape}")
