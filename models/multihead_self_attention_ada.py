import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_qkv = nn.Linear(d_model, 3 * d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, T, T)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)

        return output, attn_weights

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        qkv = self.w_qkv(x).view(batch_size, seq_len, self.num_heads, 3 * self.d_k)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        attn_out, _ = self.scaled_dot_product_attention(q, k, v, mask)

        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
        attn_out = attn_out.view(batch_size, seq_len, self.d_model)

        out = self.w_o(attn_out)

        return (gamma + 1) * out

