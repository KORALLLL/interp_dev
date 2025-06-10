import torch
import torch.nn as nn
from einops import rearrange
from models import SinusoidalPositionalEmbedding, TransformerBlockAdaLN, MlpAdaLN

class TransformerEncoderAdaLN(nn.Module):
    def __init__(self, vocab_size, C, H, W, hidden_dim=512, num_heads=8, num_blocks=6, dropout=0.1, pad_token_id=0, max_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.pos_embedding = SinusoidalPositionalEmbedding(max_len=max_len, hidden_dim=hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlockAdaLN(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.mlp_adaln = MlpAdaLN(C=C, H=H, W=W, hidden_dim=hidden_dim)

    def forward(self, token_ids, attention_mask=None, speech_embedding=None):
        """
        token_ids: [B, seq_len]
        speech_embedding: [B, C, H, W]
        """
        x = self.embedding(token_ids)  # [B, seq_len, hidden_dim]
        pos_encoding = self.pos_embedding(x.size(1)).to(x.device)
        x = x + pos_encoding.unsqueeze(0)
        x = self.dropout(x)

        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.mlp_adaln(speech_embedding)

        alpha1 = alpha1.unsqueeze(1).expand(-1, x.size(1), -1)
        beta1 = beta1.unsqueeze(1).expand_as(alpha1)
        gamma1 = gamma1.unsqueeze(1).expand_as(alpha1)
        alpha2 = alpha2.unsqueeze(1).expand_as(alpha1)
        beta2 = beta2.unsqueeze(1).expand_as(alpha1)
        gamma2 = gamma2.unsqueeze(1).expand_as(alpha1)

        for block in self.blocks:
            x = block(x, alpha1, beta1, gamma1, alpha2, beta2, gamma2, attention_mask=attention_mask)

        return x
