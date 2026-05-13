import torch
import torch.nn as nn
import torch.nn.functional as F


class SDFBoundaryHead(nn.Module):
    """
    Lightweight MLP head that predicts per-class signed distance values
    for each sphere token.
    """

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        hidden = max(1, in_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        return self.mlp(x)  # [B, N, K]

