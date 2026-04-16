import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from training.data_loader import get_batch
from training.utils import save_checkpoint, load_checkpoint, gradient_clipping
from transformer.transformer_lm import TransformerLM
from training.loss import cross_entropy_loss
from training.lr_schedule import lr_cosine_schedule
from training.optimizer import AdamW

try:
    import wandb
except ImportError:
    wandb = None


def make_memmap(path: str, dtype: str = "uint16") -> np.memmap:
    return np.memmap(path, dtype=dtype, mode="r")


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_steps: int,
) -> float:
    model.eval()
    losses = []

    for _ in range(eval_steps):
        x, y = get_batch(
            data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        losses.append(loss.item())

    model.train()
    return float(sum(losses) / len(losses))


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        theta=args.rope_theta,
    )
    return model


def get_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def maybe_save_checkpoint(
    model: torch.nn.Module,
    optimizer: AdamW,
    iteration: int,
    checkpoint_path: str | None,
) -> None:
    if checkpoint_path is None:
        return

    checkpoint_path = str(checkpoint_path)
    parent = os.path.dirname(checkpoint_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    save_checkpoint(model, optimizer, iteration, checkpoint_path)


def set_learning_rate(optimizer: AdamW, lr: float) -> None:
    if hasattr(optimizer, "param_groups"):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    elif hasattr(optimizer, "lr"):
        optimizer.lr = lr
    else:
        raise AttributeError("Optimizer does not expose param_groups or lr.")


def maybe_init_wandb(args: argparse.Namespace):
    if not args.use_wandb:
        return None

    if wandb is None:
        raise ImportError("wandb is not installed. Run `pip install wandb` or disable --use-wandb.")

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )


def train(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    print(f"Using device: {device}")

    train_data = make_memmap(args.train_data, dtype=args.data_dtype)
    val_data = make_memmap(args.val_data, dtype=args.data_dtype)

    model = build_model(args).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume_from is not None and os.path.exists(args.resume_from):
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from checkpoint at iteration {start_iter}")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    wandb_run = maybe_init_wandb(args)
    start_time = time.time()

    model.train()

    for iteration in range(start_iter, args.max_iters):
        lr = lr_cosine_schedule(
            iteration,
            args.max_lr,
            args.min_lr,
            args.warmup_iters,
            args.cosine_cycle_iters,
        )
        set_learning_rate(optimizer, lr)

        x, y = get_batch(
            train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        loss.backward()

        if args.grad_clip is not None and args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)

        optimizer.step()

        elapsed_time = time.time() - start_time

        if iteration % args.log_every == 0:
            train_loss = loss.item()
            train_ppl = math.exp(train_loss) if train_loss < 20 else float("inf")

            print(
                f"iter {iteration:6d} | "
                f"time {elapsed_time:8.2f}s | "
                f"lr {lr:.6e} | "
                f"train loss {train_loss:.4f} | "
                f"train ppl {train_ppl:.4f}"
            )

            if wandb_run is not None:
                wandb.log(
                    {
                        "step": iteration,
                        "wall_time": elapsed_time,
                        "lr": lr,
                        "train/loss": train_loss,
                        "train/ppl": train_ppl,
                    },
                    step=iteration,
                )

        if iteration % args.eval_every == 0:
            val_loss = evaluate_loss(
                model=model,
                data=val_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
                eval_steps=args.eval_steps,
            )
            val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")

            print(
                f"iter {iteration:6d} | "
                f"time {elapsed_time:8.2f}s | "
                f"val loss {val_loss:.4f} | "
                f"val ppl {val_ppl:.4f}"
            )

            if wandb_run is not None:
                wandb.log(
                    {
                        "step": iteration,
                        "wall_time": elapsed_time,
                        "val/loss": val_loss,
                        "val/ppl": val_ppl,
                    },
                    step=iteration,
                )

        if args.checkpoint_every > 0 and iteration > 0 and iteration % args.checkpoint_every == 0:
            if args.checkpoint_path is not None:
                ckpt_path = args.checkpoint_path
            else:
                ckpt_dir = Path(args.checkpoint_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = str(ckpt_dir / f"checkpoint_{iteration}.pt")

            maybe_save_checkpoint(model, optimizer, iteration, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    if args.final_checkpoint_path is not None:
        maybe_save_checkpoint(model, optimizer, args.max_iters, args.final_checkpoint_path)
        print(f"Saved final checkpoint to {args.final_checkpoint_path}")

    if wandb_run is not None:
        wandb.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument("--data-dtype", type=str, default="uint16")

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=1000)
    parser.add_argument("--cosine-cycle-iters", type=int, default=9000)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=20)

    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--final-checkpoint-path", type=str, default="checkpoints/final.pt")
    parser.add_argument("--resume-from", type=str, default=None)

    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="cs336-lm")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)