import torch.nn as nn
from models import TransformerEncoderAdaLN

class FormantPredictor(nn.Module):
    def __init__(self, vocab_size, C, H, W, hidden_dim=512, num_formants=3, pad_token_id=0, max_len=256, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoderAdaLN(
            vocab_size=vocab_size,
            C=C,
            H=H,
            W=W,
            hidden_dim=hidden_dim,
            pad_token_id=pad_token_id,
            max_len=max_len,
            dropout=dropout
        )
        self.regressor = nn.Linear(hidden_dim, num_formants)

    def forward(self, token_ids, attention_mask=None, speech_embedding=None):
        x = self.encoder(token_ids, attention_mask=attention_mask, speech_embedding=speech_embedding)
        formants = self.regressor(x)
        return formants
