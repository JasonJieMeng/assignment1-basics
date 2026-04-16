import random
from tokenizer import Tokenizer

NUM_DOCS = 10
SEED = 42

TINYSTORIES_VOCAB_PATH = "artifacts/tinystories_vocab.pkl"
TINYSTORIES_MERGES_PATH = "artifacts/tinystories_merges.pkl"

OWT_VOCAB_PATH = "artifacts/owt_vocab.pkl"
OWT_MERGES_PATH = "artifacts/owt_merges.pkl"


def sample_texts(dataset, text_key: str, n: int, seed: int = 42):
    rng = random.Random(seed)
    total = len(dataset)
    indices = rng.sample(range(total), n)
    return [dataset[i][text_key] for i in indices]


def compute_bytes_per_token(tokenizer: Tokenizer, texts: list[str]) -> float:
    total_bytes = 0
    total_tokens = 0

    for text in texts:
        text_bytes = text.encode("utf-8")
        ids = tokenizer.encode(text)

        total_bytes += len(text_bytes)
        total_tokens += len(ids)

    if total_tokens == 0:
        raise ValueError("Total token count is zero.")

    return total_bytes / total_tokens


def summarize_tokenizer_on_corpus(
    tokenizer: Tokenizer,
    texts: list[str],
    tokenizer_name: str,
    corpus_name: str,
):
    total_bytes = 0
    total_tokens = 0

    for text in texts:
        text_bytes = text.encode("utf-8")
        ids = tokenizer.encode(text)

        total_bytes += len(text_bytes)
        total_tokens += len(ids)

    ratio = total_bytes / total_tokens

    print(f"{tokenizer_name} on {corpus_name}")
    print(f"  Total bytes : {total_bytes}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Bytes/token : {ratio:.4f}")
    print()


def sample_docs_from_txt(filepath, n=10, seed=42):
    random.seed(seed)

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    return random.sample(lines, n)


def main():
    tiny_tokenizer = Tokenizer.from_files(
        vocab_filepath=TINYSTORIES_VOCAB_PATH,
        merges_filepath=TINYSTORIES_MERGES_PATH,
        special_tokens=["<|endoftext|>"],
    )

    owt_tokenizer = Tokenizer.from_files(
        vocab_filepath=OWT_VOCAB_PATH,
        merges_filepath=OWT_MERGES_PATH,
        special_tokens=["<|endoftext|>"],
    )

    tiny_docs = sample_docs_from_txt("../data/TinyStoriesV2-GPT4-train.txt", n=10)
    owt_docs = sample_docs_from_txt("../data/owt_train.txt", n=10)
    

    print("=" * 60)
    print("Sampled document counts")
    print(f"TinyStories: {len(tiny_docs)}")
    print(f"OpenWebText: {len(owt_docs)}")
    print("=" * 60)
    print()

    summarize_tokenizer_on_corpus(
        tokenizer=tiny_tokenizer,
        texts=tiny_docs,
        tokenizer_name="TinyStories tokenizer",
        corpus_name="TinyStories samples",
    )

    summarize_tokenizer_on_corpus(
        tokenizer=owt_tokenizer,
        texts=tiny_docs,
        tokenizer_name="OpenWebText tokenizer",
        corpus_name="TinyStories samples",
    )

    summarize_tokenizer_on_corpus(
        tokenizer=tiny_tokenizer,
        texts=owt_docs,
        tokenizer_name="TinyStories tokenizer",
        corpus_name="OpenWebText samples",
    )

    summarize_tokenizer_on_corpus(
        tokenizer=owt_tokenizer,
        texts=owt_docs,
        tokenizer_name="OpenWebText tokenizer",
        corpus_name="OpenWebText samples",
    )

    tiny_ratio = compute_bytes_per_token(tiny_tokenizer, tiny_docs)
    owt_ratio = compute_bytes_per_token(owt_tokenizer, owt_docs)

    print("=" * 60)
    print(f"TinyStories tokenizer on TinyStories docs: {tiny_ratio:.4f} bytes/token")
    print(f"OpenWebText tokenizer on OpenWebText docs: {owt_ratio:.4f} bytes/token")
    print("=" * 60)


if __name__ == "__main__":
    main()