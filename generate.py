import argparse
from pathlib import Path

import torch

from inference.decode import decode
from cs336_basics.tokenizer import Tokenizer
from transformer.transformer_lm import TransformerLM


def load_model(checkpoint_path: str, device: torch.device) -> tuple[torch.nn.Module, dict]:

    model_config = {
        "vocab_size": 32000,
        "context_length": 512,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
        "theta": 10000.0,
    }

    model = TransformerLM(**model_config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, model_config


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained owt checkpoint.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--vocab-path", type=str, required=True)
    parser.add_argument("--merges-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output-path", type=str, default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    vocab_path = Path(args.vocab_path)
    merges_path = Path(args.merges_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    if not merges_path.exists():
        raise FileNotFoundError(f"Merges file not found: {merges_path}")

    device = torch.device(args.device)

    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(vocab_path),
        merges_filepath=str(merges_path),
        special_tokens=["<|endoftext|>"],
    )

    model, model_config = load_model(str(checkpoint_path), device)

    prompt_ids = tokenizer.encode(args.prompt)
    if len(prompt_ids) == 0:
        raise ValueError("Prompt encoded to an empty sequence.")

    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    eos_token_id = tokenizer.token_to_id.get(b"<|endoftext|>", None)

    output_ids = decode(
        model=model,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        context_length=model_config["context_length"],
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
    )

    generated_text = tokenizer.decode(output_ids[0].tolist())
    print(generated_text)

    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.write_text(generated_text, encoding="utf-8")
        print(f"\nSaved generated text to {output_path}")


if __name__ == "__main__":
    main()