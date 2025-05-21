import torch
import torch.nn as nn
import math

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_len, hidden_dim, device=None):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float) * (-math.log(10000.0) / hidden_dim))

        pe = torch.zeros(max_len, hidden_dim, dtype=torch.float)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, seq_len):
        return self.pe[:seq_len]
