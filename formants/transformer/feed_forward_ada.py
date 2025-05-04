import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardAda(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        ff_out = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        return gamma * ff_out