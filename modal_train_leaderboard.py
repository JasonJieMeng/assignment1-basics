import modal

app = modal.App("cs336-owt-train-leaderboard")

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

volume = modal.Volume.from_name("cs336-owt", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb-secret")


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60 * 2,
    volumes={"/vol": volume},
    secrets=[wandb_secret],
)
def train_owt_model():
    import subprocess

    cmd = [
        "python",
        "-m",
        "training.leaderboard_train",
        "--train-data", "/vol/data/owt_train.bin",
        "--val-data", "/vol/data/owt_valid.bin",
        "--data-dtype", "uint16",
        "--vocab-size", "32000",
        "--context-length", "512",
        "--d-model", "512",
        "--d-ff", "1344",
        "--num-layers", "4",
        "--num-heads", "16",
        "--rope-theta", "10000",
        "--batch-size", "64",
        "--max-iters", "20000",
        "--max-lr", "3e-4",
        "--min-lr", "3e-5",
        "--warmup-iters", "2000",
        "--cosine-cycle-iters", "38000",
        "--beta1", "0.9",
        "--beta2", "0.95",
        "--eps", "1e-8",
        "--weight-decay", "0.1",
        "--grad-clip", "1.0",
        "--eval-every", "200",
        "--eval-steps", "20",
        "--log-every", "10",
        "--checkpoint-every", "1000",
        "--checkpoint-dir", "/vol/checkpoints/owt_leaderboard",
        "--final-checkpoint-path", "/vol/checkpoints/owt_leaderboard/final.pt",
        "--device", "cuda",
        "--use-wandb",
        "--wandb-project", "cs336-lm",
        "--wandb-run-name", "owt-leaderboard",
        "--max-seconds", "2640" # 44 mins
    ]

    print("Running command:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)
    volume.commit()


@app.local_entrypoint()
def main():
    train_owt_model.remote()