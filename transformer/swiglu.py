import math
import torch
from torch import nn

from .linear import Linear


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        if d_ff is None:
            d_ff = math.ceil(((8 / 3) * d_model) / 64) * 64
        self.d_ff = d_ff

        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x3 = self.w3(x)

        silu_x1 = x1 * torch.sigmoid(x1)
        return self.w2(silu_x1 * x3)