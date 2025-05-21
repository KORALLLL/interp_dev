import torch
import torch.nn as nn
from models.transformer_Block_AdaLN import TransformerBlockAdaLN
from models.positional_encoding import SinusoidalPositionalEmbedding

class TransformerEncoderAdaLN(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, num_heads=8, num_blocks=6, dropout=0.1, pad_token_id=0, max_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.pos_embedding = SinusoidalPositionalEmbedding(max_len=max_len, hidden_dim=hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlockAdaLN(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, token_ids, attention_mask=None, adaln_params=None):
        if adaln_params is None:
            raise ValueError("adaln_params must be provided to TransformerEncoderAdaLN.")

        x = self.embedding(token_ids)
        pos_encoding = self.pos_embedding(x.size(1)).to(x.device)
        x = x + pos_encoding.unsqueeze(0)
        x = self.dropout(x)

        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = adaln_params

        alpha1 = alpha1.unsqueeze(1).expand_as(x)
        beta1 = beta1.unsqueeze(1).expand_as(x)
        gamma1 = gamma1.unsqueeze(1).expand_as(x)
        alpha2 = alpha2.unsqueeze(1).expand_as(x)
        beta2 = beta2.unsqueeze(1).expand_as(x)
        gamma2 = gamma2.unsqueeze(1).expand_as(x)

        for block in self.blocks:
            x = block(x, alpha1, beta1, gamma1, alpha2, beta2, gamma2, attention_mask=attention_mask)

        return x
