import modal

app = modal.App("cs336-tinystories-train")

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

@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60 * 1,
    volumes={"/vol": volume},
    secrets=[wandb_secret],
)
def train_base_model():
    import subprocess

    cmd = [
        "python",
        "-m",
        "training.train",
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
        "--batch-size", "32",
        "--max-iters", "40000",
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
        "--checkpoint-dir", "/vol/checkpoints",
        "--final-checkpoint-path", "/vol/checkpoints/final.pt",
        "--device", "cuda",
        "--use-wandb",
        "--wandb-project", "cs336-lm",
        "--wandb-run-name", "tinystories-base",
    ]

    subprocess.run(cmd, check=True)
    volume.commit()


@app.local_entrypoint()
def main():
    train_base_model.remote()