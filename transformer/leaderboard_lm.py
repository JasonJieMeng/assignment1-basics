import torch
from torch import nn

from .embedding import Embedding
from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        theta: float,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.theta = theta
        self.eps = eps

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=theta,
                    eps=eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        # NOTE: no separate lm_head. We tie weights with token_embeddings.
        self._init_weights()

    def _init_weights(self) -> None:
        # Embedding init: std = 1/sqrt(d_model). Small enough that the tied
        # output projection produces reasonable initial logits (~uniform over vocab).
        emb_std = self.d_model ** -0.5
        nn.init.trunc_normal_(
            self.token_embeddings.weight,
            mean=0.0,
            std=emb_std,
            a=-3 * emb_std,
            b=3 * emb_std,
        )

        # Scale down residual-path output projections by 1/sqrt(2 * num_layers)
        # to keep residual-stream variance bounded with depth (GPT-2 trick).
        residual_std = (2.0 * self.num_layers) ** -0.5
        for layer in self.layers:
            for W in (layer.attn.W_o.W, layer.ffn.w2.W):
                nn.init.trunc_normal_(
                    W,
                    mean=0.0,
                    std=residual_std,
                    a=-3 * residual_std,
                    b=3 * residual_std,
                )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = token_ids.shape
        token_positions = torch.arange(seq_len, device=token_ids.device)

        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x = layer(x, token_positions)

        x = self.ln_final(x)
        # Tied output projection: token_embeddings.weight is (vocab_size, d_model),
        # so we transpose to get (d_model, vocab_size).
        logits = x @ self.token_embeddings.weight.T
        return logits