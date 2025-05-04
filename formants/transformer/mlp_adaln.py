import torch
import torch.nn as nn

class MlpAdaLN(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 6 * hidden_dim)

    def forward(self, x):
        out = self.fc2(self.relu(self.fc1(x)))
        return torch.chunk(out, 6, dim=-1)  


