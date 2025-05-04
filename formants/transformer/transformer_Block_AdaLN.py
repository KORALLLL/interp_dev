import torch.nn as nn
from formants.transformer.layer_norm_ada import AdaptiveLayerNorm
from formants.transformer.multihead_self_attention_ada import MultiHeadSelfAttention
from formants.transformer.feed_forward_ada import FeedForwardAda

class TransformerBlockAdaLN(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = AdaptiveLayerNorm(hidden_dim)
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = AdaptiveLayerNorm(hidden_dim)
        self.ff = FeedForwardAda(hidden_dim, d_ff=4 * hidden_dim, dropout=dropout)

    def forward(self, x, alpha1, beta1, gamma1, alpha2, beta2, gamma2):
        x_norm = self.norm1(x, alpha1, beta1)
        x = x + self.attn(x_norm, gamma1)

        x_norm = self.norm2(x, alpha2, beta2)
        x = x + self.ff(x_norm, gamma2)

        return x
