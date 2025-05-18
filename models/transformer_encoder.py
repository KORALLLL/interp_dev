import torch
import torch.nn as nn
from formants.transformer.mlp_adaln import MlpAdaLN
from formants.transformer.transformer_Block_AdaLN import TransformerBlockAdaLN
from formants.transformer.positional_encoding import SinusoidalPosEmbedding

class TransformerEncoderAdaLN(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, num_heads=8, num_blocks = 6, dropout=0.1, pad_token_id=0, max_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlockAdaLN(hidden_dim=hidden_dim,num_heads=num_heads,dropout=0.1) for i in range(num_blocks)])
        self.param_generator = MlpAdaLN(input_dim=256, hidden_dim=hidden_dim)

    def forward(self, token_ids, speech_embedding, attention_mask=None):
        x = self.embedding(token_ids)
        pos_encoding = SinusoidalPosEmbedding(max_len=x.size(1), hidden_dim=self.hidden_dim, device=x.device)
        x = x + pos_encoding.unsqueeze(0)
        x = self.dropout(x)
        for block in self.blocks:
            alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.param_generator(speech_embedding)

            alpha1 = alpha1.unsqueeze(1).expand_as(x)
            beta1 = beta1.unsqueeze(1).expand_as(x)
            gamma1 = gamma1.unsqueeze(1).expand_as(x)
            alpha2 = alpha2.unsqueeze(1).expand_as(x)
            beta2 = beta2.unsqueeze(1).expand_as(x)
            gamma2 = gamma2.unsqueeze(1).expand_as(x)

            x = block(x, alpha1, beta1, gamma1, alpha2, beta2, gamma2,
                      attention_mask=attention_mask)

        return x

