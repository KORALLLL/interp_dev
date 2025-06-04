import torch
import torch.nn as nn
from models import SinusoidalPositionalEmbedding,TransformerBlockAdaLN,MlpAdaLN
class TransformerEncoderAdaLN(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim=512, num_heads=8, num_blocks=6, dropout=0.1, pad_token_id=0, max_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.pos_embedding = SinusoidalPositionalEmbedding(max_len=max_len, hidden_dim=hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlockAdaLN(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.mlp_adaln = MlpAdaLN(input_dim=input_dim, hidden_dim=hidden_dim)

    def forward(self, token_ids, attention_mask=None, mlp_input=None):


        x = self.embedding(token_ids)
        pos_encoding = self.pos_embedding(x.size(1)).to(x.device)
        x = x + pos_encoding.unsqueeze(0)
        x = self.dropout(x)

        # Используем встроенный MLP для получения параметров
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.mlp_adaln(mlp_input)

        # Расширяем по длине последовательности
        alpha1 = alpha1.unsqueeze(1).expand(-1, x.size(1), -1)
        beta1 = beta1.unsqueeze(1).expand(-1, x.size(1), -1)
        gamma1 = gamma1.unsqueeze(1).expand(-1, x.size(1), -1)
        alpha2 = alpha2.unsqueeze(1).expand(-1, x.size(1), -1)
        beta2 = beta2.unsqueeze(1).expand(-1, x.size(1), -1)
        gamma2 = gamma2.unsqueeze(1).expand(-1, x.size(1), -1)

        for block in self.blocks:
            x = block(x, alpha1, beta1, gamma1, alpha2, beta2, gamma2, attention_mask=attention_mask)

        return x

