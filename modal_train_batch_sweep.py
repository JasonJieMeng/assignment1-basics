import json
import modal

app = modal.App("cs336-tinystories-batch-sweep")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "wandb",
        "einops",
    )
    .add_local_python_source("training", "transformer", "cs336_basics")
)

volume = modal.Volume.from_name("cs336-tinystories", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb-secret")

BATCH_SIZES = [1, 8, 32, 64, 128, 256, 512]

FIXED_MAX_LR = 1e-3
FIXED_MIN_LR = FIXED_MAX_LR / 10.0

TOTAL_TOKENS = 327_680_000
CONTEXT_LENGTH = 256

BASE_ARGS = [
    "python",
    "-m",
    "training.train",
    "--train-data", "/vol/data/tiny_train.bin",
    "--val-data", "/vol/data/tiny_valid.bin",
    "--data-dtype", "uint16",
    "--vocab-size", "10000",
    "--context-length", str(CONTEXT_LENGTH),
    "--d-model", "512",
    "--d-ff", "1344",
    "--num-layers", "4",
    "--num-heads", "16",
    "--rope-theta", "10000",
    "--beta1", "0.9",
    "--beta2", "0.95",
    "--eps", "1e-8",
    "--weight-decay", "0.1",
    "--grad-clip", "1.0",
    "--eval-every", "200",
    "--eval-steps", "20",
    "--log-every", "10",
    "--checkpoint-every", "1000",
    "--device", "cuda",
    "--use-wandb",
    "--wandb-project", "cs336-lm",
]


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60,
    volumes={"/vol": volume},
    secrets=[wandb_secret],
)
def train_one_batch_size(batch_size: int):
    import subprocess
    from pathlib import Path

    max_iters = TOTAL_TOKENS // (batch_size * CONTEXT_LENGTH)

    warmup_iters = max_iters // 20          # 5%
    cosine_cycle_iters = max_iters - warmup_iters

    run_name = f"tinystories-batch-{batch_size}"
    ckpt_dir = f"/vol/checkpoints/{run_name}"
    final_ckpt = f"{ckpt_dir}/final.pt"

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    cmd = BASE_ARGS + [
        "--batch-size", str(batch_size),
        "--max-iters", str(max_iters),
        "--warmup-iters", str(warmup_iters),
        "--cosine-cycle-iters", str(cosine_cycle_iters),
        "--max-lr", str(FIXED_MAX_LR),
        "--min-lr", str(FIXED_MIN_LR),
        "--checkpoint-dir", ckpt_dir,
        "--final-checkpoint-path", final_ckpt,
        "--wandb-run-name", run_name,
    ]

    print(f"\n=== Starting run: {run_name} ===")
    print(f"Batch size: {batch_size}")
    print(f"Max iters: {max_iters}")
    print(f"Total tokens: {batch_size * max_iters * CONTEXT_LENGTH:,}")
    print("Command:", " ".join(cmd))

    subprocess.run(cmd, check=True)

    summary_path = Path(ckpt_dir) / "sweep_config.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "run_name": run_name,
                "batch_size": batch_size,
                "max_iters": max_iters,
                "total_tokens": batch_size * max_iters * CONTEXT_LENGTH,
                "max_lr": FIXED_MAX_LR,
                "min_lr": FIXED_MIN_LR,
                "checkpoint_dir": ckpt_dir,
                "final_checkpoint_path": final_ckpt,
            },
            f,
            indent=2,
        )

    volume.commit()
    return {
        "run_name": run_name,
        "batch_size": batch_size,
        "max_iters": max_iters,
        "total_tokens": batch_size * max_iters * CONTEXT_LENGTH,
    }


@app.local_entrypoint()
def main(parallel: bool = True):
    if parallel:
        results = list(train_one_batch_size.map(BATCH_SIZES))
    else:
        results = [train_one_batch_size.remote(bs) for bs in BATCH_SIZES]

    print("\nFinished sweep:")
    for result in results:
        print(result)