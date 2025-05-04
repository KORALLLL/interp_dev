import torch
import torch.nn as nn

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len: int, hidden_dim: int):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        return x + pos_emb
