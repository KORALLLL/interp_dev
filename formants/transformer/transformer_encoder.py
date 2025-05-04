import torch
import torch.nn as nn
from formants.transformer.mlp_adaln import MlpAdaLN
from formants.transformer.transformer_Block_AdaLN import TransformerBlockAdaLN
from formants.transformer.positional_encoding import LearnablePositionalEncoding

class TransformerEncoderAdaLN(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, num_heads=8, dropout=0.1, pad_token_id=0, max_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.pos_encoding = LearnablePositionalEncoding(max_len=max_len, hidden_dim=hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.block = TransformerBlockAdaLN(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.param_generator = MlpAdaLN(input_dim=256, hidden_dim=hidden_dim)

    def forward(self, token_ids, speech_embedding):

        x = self.embedding(token_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.param_generator(speech_embedding)  # (B, H)

        alpha1 = alpha1.unsqueeze(1).expand_as(x)
        beta1  = beta1.unsqueeze(1).expand_as(x)
        gamma1 = gamma1.unsqueeze(1).expand_as(x)
        alpha2 = alpha2.unsqueeze(1).expand_as(x)
        beta2  = beta2.unsqueeze(1).expand_as(x)
        gamma2 = gamma2.unsqueeze(1).expand_as(x)

        return self.block(x, alpha1, beta1, gamma1, alpha2, beta2, gamma2)
