import torch
import torch.nn as nn
from einops import rearrange


class MlpAdaLN(nn.Module):
    def __init__(self, C, H, W, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(C, 1)
        self.fc2 = nn.Linear(H, 6)
        self.fc3 = nn.Linear(W, hidden_dim)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        debug = x.shape[0] <= 3

        if debug:
            print(f"\n[MlpAdaLN] Input x shape: {x.shape}")
            print(f"  x stats: mean={x.mean().item():.4f}, std={x.std().item():.4f}")

        # [B, C, H, W] → [B, H, W, C]
        x1 = rearrange(x, 'b c h w -> b h w c')
        if debug:
            print(f"→ fc1 input: {x1.shape} | applying fc1 (C → 1)")

        x1 = self.fc1(x1)  # → [B, H, W, 1]
        if debug:
            print(f"  fc1 output: {x1.shape}, mean={x1.mean().item():.4f}, std={x1.std().item():.4f}")

        x1 = rearrange(x1, 'b h w 1 -> b h w')

        # [B, H, W] → [B, W, H]
        x2 = rearrange(x1, 'b h w -> b w h')
        if debug:
            print(f"→ fc2 input: {x2.shape} | applying fc2 (H → 6)")

        x2 = self.fc2(x2)  # → [B, W, 6]
        if debug:
            print(f"  fc2 output: {x2.shape}, mean={x2.mean().item():.4f}, std={x2.std().item():.4f}")

        # [B, W, 6] → [B, 6, W]
        x3 = rearrange(x2, 'b w six -> b six w', six=6)
        if debug:
            print(f"→ fc3 input: {x3.shape} | applying fc3 (W → hidden_dim)")

        x3 = self.fc3(x3)  # → [B, 6, hidden_dim]
        if debug:
            print(f"  fc3 output: {x3.shape}, mean={x3.mean().item():.4f}, std={x3.std().item():.4f}")

        alphas_betas_gammas = torch.chunk(x3, chunks=6, dim=1)
        return [t.squeeze(1) for t in alphas_betas_gammas]

