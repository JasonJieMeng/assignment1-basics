import json
import modal

app = modal.App("cs336-tinystories-abla-nope")

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

LR_SWEEP = [1e-3]

BASE_ARGS = [
    "python",
    "-m",
    "training.nope_train",
    "--train-data", "/vol/data/tiny_train.bin",
    "--val-data", "/vol/data/tiny_valid.bin",
    "--data-dtype", "uint16",
    "--vocab-size", "10000",
    "--context-length", "256",
    "--d-model", "512",
    "--d-ff", "1344",
    "--num-layers", "4",
    "--num-heads", "16",
    "--rope-theta", "10000",
    "--batch-size", "128",
    "--max-iters", "10000",
    "--warmup-iters", "500",
    "--cosine-cycle-iters", "9500",
    "--beta1", "0.9",
    "--beta2", "0.95",
    "--eps", "1e-8",
    "--weight-decay", "0.1",
    "--grad-clip", "0",
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
def train_one_lr(max_lr: float):
    import subprocess
    from pathlib import Path

    min_lr = max_lr / 10.0
    lr_tag = f"{max_lr:.0e}".replace("+", "")
    run_name = f"tinystories-nope-{lr_tag}"
    ckpt_dir = f"/vol/checkpoints/{run_name}"
    final_ckpt = f"{ckpt_dir}/final.pt"

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    cmd = BASE_ARGS + [
        "--max-lr", str(max_lr),
        "--min-lr", str(min_lr),
        "--checkpoint-dir", ckpt_dir,
        "--final-checkpoint-path", final_ckpt,
        "--wandb-run-name", run_name,
    ]

    print(f"Starting run: {run_name}")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    summary_path = Path(ckpt_dir) / "sweep_config.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "run_name": run_name,
                "max_lr": max_lr,
                "min_lr": min_lr,
                "checkpoint_dir": ckpt_dir,
                "final_checkpoint_path": final_ckpt,
            },
            f,
            indent=2,
        )

    volume.commit()
    return {
        "run_name": run_name,
        "max_lr": max_lr,
        "min_lr": min_lr,
        "checkpoint_dir": ckpt_dir,
        "final_checkpoint_path": final_ckpt,
    }


@app.local_entrypoint()
def main(parallel: bool = True):
    if parallel:
        results = list(train_one_lr.map(LR_SWEEP))
    else:
        results = [train_one_lr.remote(lr) for lr in LR_SWEEP]

    print("\nFinished sweep:")
    for result in results:
        print(result)
