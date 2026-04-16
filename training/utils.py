import torch
from torch import Tensor
from typing import Iterable
import os
import typing

def gradient_clipping(parameters: Iterable[Tensor], max_norm: float, eps: float = 1e-6) -> None:

    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return

    total_norm_sq = torch.sum(
        torch.stack([torch.sum(g ** 2) for g in grads])
    )
    total_norm = torch.sqrt(total_norm_sq)

    clip_coef = max_norm / (total_norm + eps)

    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]