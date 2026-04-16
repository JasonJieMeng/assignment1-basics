import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ) -> None:
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, got {d_k}")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        k = torch.arange(0, d_k // 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (2 * k / d_k))

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        angles = torch.outer(positions, inv_freq)

        cos = torch.repeat_interleave(torch.cos(angles), repeats=2, dim=-1)
        sin = torch.repeat_interleave(torch.sin(angles), repeats=2, dim=-1)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:

        if x.shape[-1] != self.d_k:
            raise ValueError(
                f"Expected x.shape[-1] == {self.d_k}, got {x.shape[-1]}"
            )

        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        x_rot = torch.empty_like(x)
        x_rot[..., ::2] = -x_odd
        x_rot[..., 1::2] = x_even

        return x * cos + x_rot * sin