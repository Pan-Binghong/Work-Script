#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Decoder.py
@Time    :   2024/09/29 09:57:49
@Author  :   pan binghong 
@Email   :   19909442097@163.com
@description   :   
'''
import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from FeedForwardNetwork import FeedForwardNetwork
from ResidualConnection import ResidualConnection
from LayerNormalization import LayerNormalization

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, hidden_size, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadAttention(d_model, heads)
        self.src_attn = MultiHeadAttention(d_model, heads)
        self.feed_forward = FeedForwardNetwork(d_model, hidden_size, dropout)
        self.residuals = nn.ModuleList([
            ResidualConnection(d_model, dropout),
            ResidualConnection(d_model, dropout),            
            ResidualConnection(d_model, dropout)
        ])

    def forward(self, x, memory, src_mask=None, trg_mask=None):
        # Self-attention
        x = self.residuals[0](x, lambda x: self.self_attn(x, x, x, trg_mask))
        # Encoder-decoder attention
        x = self.residuals[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        # Feed-forward network
        return self.residuals[2](x, self.feed_forward)



class Decoder(nn.Module):
    def __init__(self, num_layers, decoder_layer):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.norm = LayerNormalization(decoder_layer.d_model)

    def forward(self, trg, memory, trg_mask=None, src_mask=None):
            for layer in self.layers:
                trg = layer(trg, memory, trg_mask, src_mask)
            return self.norm(trg)


# Testing Decoder and DecoderLayer
if __name__ == '__main__':
    # Model parameters
    d_model = 512
    heads = 8
    hidden_size = 2048
    num_layers = 6
    dropout = 0.1
    batch_size = 32
    seq_len_trg = 10  # Length of target sequence
    seq_len_src = 12  # Length of source sequence

    # Create a single decoder layer and the decoder
    decoder_layer = DecoderLayer(d_model, heads, hidden_size, dropout)
    decoder = Decoder(num_layers, decoder_layer)

    # Generate random target (trg) and memory (encoder output) for testing
    trg = torch.randn((batch_size, seq_len_trg, d_model))  # Target sequence
    memory = torch.randn((batch_size, seq_len_src, d_model))  # Encoder output (source sequence)

    # Create masks
    trg_mask = torch.ones((batch_size, seq_len_trg, seq_len_trg))  # Mask for target sequence
    trg_mask[:, :, 5:] = 0  # Example: mask out tokens after position 5
    src_mask = torch.ones((batch_size, seq_len_src))  # Mask for source sequence
    src_mask[:, 8:] = 0  # Example: mask out tokens after position 8

    # Adjust masks shape for multi-head attention
    trg_mask = trg_mask.unsqueeze(1).unsqueeze(1)  # Shape: (batch_size, 1, 1, seq_len_trg, seq_len_trg)
    src_mask = src_mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len_src)

    # Forward pass through the decoder
    output = decoder(trg, memory, trg_mask=trg_mask, src_mask=src_mask)

    # Print output shape
    print(f"Decoder output shape: {output.shape}")