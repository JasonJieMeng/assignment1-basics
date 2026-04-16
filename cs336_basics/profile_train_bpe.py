import cProfile
import pstats

from train_bpe import train_bpe

def main():
    profiler = cProfile.Profile()
    profiler.enable()

    vocab, merges = train_bpe(
        input_path="../data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        show_progress=True,
        num_processes=3,
        num_chunks=12,
    )

    profiler.disable()
    profiler.dump_stats("bpe.prof")

    stats = pstats.Stats("bpe.prof")
    stats.strip_dirs().sort_stats("cumulative").print_stats(30)

    print(f"Learned vocab size: {len(vocab)}")
    print(f"Num merges: {len(merges)}")

if __name__ == "__main__":
    main()