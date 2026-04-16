import numpy as np
import torch
from torch import Tensor

def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:

    if len(x.shape) != 1:
        raise ValueError("x must be a 1D numpy array")
    if len(x) < context_length + 1:
        raise ValueError("x is too short for the requested context_length")

    # valid starts are 0 through len(x) - context_length - 1 inclusive
    starts = np.random.randint(0, len(x) - context_length, size=batch_size)

    inputs = np.stack([x[s : s + context_length] for s in starts])
    targets = np.stack([x[s + 1 : s + 1 + context_length] for s in starts])

    inputs_t = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_t = torch.tensor(targets, dtype=torch.long, device=device)
    return inputs_t, targets_t