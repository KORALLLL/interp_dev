import torch
import torch.nn as nn

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.hidden_dim = hidden_dim
        self.alph = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.ones(hidden_dim))
    def forward(self, x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        norm_x = self.alph*(x - mean) / (std + self.eps) + self.bias

        return (alpha + 1) * norm_x + beta
9