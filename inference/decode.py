import torch
from torch import Tensor
from transformer.softmax import softmax

def _sample_top_p(probs: Tensor, top_p: float) -> Tensor:

    if top_p >= 1.0:
        return torch.multinomial(probs, num_samples=1)

    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens after the nucleus threshold.
    sorted_mask = cumulative_probs > top_p
    # Keep at least the first token.
    sorted_mask[..., 0] = False

    filtered_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    sampled_sorted_idx = torch.multinomial(filtered_probs, num_samples=1)
    next_token = torch.gather(sorted_indices, dim=-1, index=sampled_sorted_idx)
    return next_token


@torch.no_grad()
def decode(
    model,
    prompt: Tensor,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> Tensor:
    
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not (0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")

    device = prompt.device
    tokens = prompt
    batch_size = prompt.shape[0]

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        idx = tokens[:, -context_length:]

        logits = model(idx)  # (B, T, V)
        next_token_logits = logits[:, -1, :]  # (B, V)

        if temperature < 1e-8:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            scaled_logits = next_token_logits / temperature
            probs = softmax(scaled_logits, dim=-1)
            next_token = _sample_top_p(probs, top_p=top_p)

        if eos_token_id is not None:
            # Once a sequence has finished, keep appending EOS.
            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, eos_token_id),
                next_token,
            )
            finished = finished | (next_token.squeeze(-1) == eos_token_id)

        tokens = torch.cat([tokens, next_token], dim=1)

        if eos_token_id is not None and torch.all(finished):
            break

    return tokens