import time
from tokenizer import Tokenizer

VOCAB_PATH = "artifacts/owt_vocab.pkl"         
MERGES_PATH = "artifacts/owt_merges.pkl"        
TEXT_PATH = "../data/owt_train.txt"      
NUM_CHARS = 5_000_000              


def benchmark_tokenizer(tokenizer, text, num_trials=3):
    text_bytes = len(text.encode("utf-8"))
    best_time = float("inf")
    best_num_tokens = None

    for _ in range(num_trials):
        start = time.perf_counter()
        ids = tokenizer.encode(text)
        elapsed = time.perf_counter() - start

        if elapsed < best_time:
            best_time = elapsed
            best_num_tokens = len(ids)

    bytes_per_second = text_bytes / best_time
    tokens_per_second = best_num_tokens / best_time
    return text_bytes, best_num_tokens, best_time, bytes_per_second, tokens_per_second


def main():
    tokenizer = Tokenizer.from_files(
        vocab_filepath=VOCAB_PATH,
        merges_filepath=MERGES_PATH,
        special_tokens=["<|endoftext|>"],
    )

    with open(TEXT_PATH, "r", encoding="utf-8") as f:
        text = f.read(NUM_CHARS)

    text_bytes, num_tokens, elapsed, bytes_per_second, tokens_per_second = benchmark_tokenizer(
        tokenizer, text, num_trials=3
    )

    pile_bytes_decimal = 825 * 10**9
    pile_bytes_binary = 825 * 1024**3  

    est_seconds_decimal = pile_bytes_decimal / bytes_per_second
    est_hours_decimal = est_seconds_decimal / 3600
    est_days_decimal = est_hours_decimal / 24

    est_seconds_binary = pile_bytes_binary / bytes_per_second
    est_hours_binary = est_seconds_binary / 3600
    est_days_binary = est_hours_binary / 24

    print(f"Benchmarked bytes          : {text_bytes}")
    print(f"Benchmarked tokens         : {num_tokens}")
    print(f"Best elapsed time          : {elapsed:.4f} s")
    print(f"Throughput                 : {bytes_per_second:.2f} bytes/s")
    print(f"Throughput                 : {tokens_per_second:.2f} tokens/s")
    print()
    print("Estimated time for 825 GB:")
    print(f"  Using decimal GB (825 * 10^9 bytes):")
    print(f"    {est_seconds_decimal:.2f} s")
    print(f"    {est_hours_decimal:.2f} hours")
    print(f"    {est_days_decimal:.2f} days")
    print()
    print(f"  Using binary GiB-style (825 * 1024^3 bytes):")
    print(f"    {est_seconds_binary:.2f} s")
    print(f"    {est_hours_binary:.2f} hours")
    print(f"    {est_days_binary:.2f} days")


if __name__ == "__main__":
    main()