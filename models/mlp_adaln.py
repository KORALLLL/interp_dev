import torch
import torch.nn as nn

class MlpAdaLN(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.fc = nn.Linear(input_dim, 6 * hidden_dim)


       #nn.init.zeros_(self.fc.weight)
       #nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        out = self.fc(x)
        return torch.chunk(out, 6, dim=-1)
