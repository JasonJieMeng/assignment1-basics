import os
import time
import json
import pickle
import resource

from train_bpe import train_bpe

INPUT_PATH = "../data/TinyStoriesV2-GPT4-train.txt"
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]

def get_peak_memory_mb() -> float:
    # On macOS ru_maxrss is in bytes; on Linux it's usually KB.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return rss / (1024 * 1024)
    return rss / 1024

def main():
    start = time.time()

    vocab, merges = train_bpe(
        input_path=INPUT_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )

    elapsed = time.time() - start
    peak_mem_mb = get_peak_memory_mb()

    os.makedirs("artifacts", exist_ok=True)

    with open("artifacts/tinystories_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("artifacts/tinystories_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    # Human-readable version too
    with open("artifacts/tinystories_vocab.json", "w", encoding="utf-8") as f:
        json.dump(
            {str(k): list(v) for k, v in vocab.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open("artifacts/tinystories_merges.txt", "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a!r}\t{b!r}\n")

    longest_token = max(vocab.values(), key=len)

    print(f"Vocab size: {len(vocab)}")
    print(f"Num merges: {len(merges)}")
    print(f"Training time: {elapsed:.2f} seconds")
    print(f"Peak memory: {peak_mem_mb:.2f} MB")
    print(f"Longest token length: {len(longest_token)}")
    print(f"Longest token bytes: {longest_token!r}")

if __name__ == "__main__":
    main()