import torch
from torch import Tensor


def cross_entropy_loss(
    logits: Tensor,
    targets: Tensor,
) -> Tensor:

    if logits.ndim < 1:
        raise ValueError("logits must have at least 1 dimension")

    if targets.shape != logits.shape[:-1]:
        raise ValueError(
            f"targets shape {targets.shape} must match logits.shape[:-1] {logits.shape[:-1]}"
        )

    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    shifted_logits = logits - max_logits

    logsumexp = max_logits.squeeze(-1) + torch.log(
        torch.sum(torch.exp(shifted_logits), dim=-1)
    )

    target_logits = torch.gather(
        logits, dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)

    loss = logsumexp - target_logits

    return loss.mean()