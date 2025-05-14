import torch
import math

def SinusoidalPosEmbedding(max_len, hidden_dim, device=None):
    position = torch.arange(max_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / hidden_dim))

    pe = torch.zeros(max_len, hidden_dim, dtype=torch.float, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe
