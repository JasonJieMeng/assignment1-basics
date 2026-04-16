import os
import multiprocessing as mp
from itertools import islice
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer


_WORKER_TOKENIZER = None


def _init_worker(vocab_path: str, merges_path: str):
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = Tokenizer.from_files(vocab_path, merges_path)


def _encode_chunk(text_chunk: str) -> np.ndarray:
    global _WORKER_TOKENIZER
    ids = _WORKER_TOKENIZER.encode(text_chunk)
    return np.asarray(ids, dtype=np.uint16)


def _batched_lines(path: str, lines_per_chunk: int):
    with open(path, "r", encoding="utf-8") as f:
        while True:
            batch = list(islice(f, lines_per_chunk))
            if not batch:
                break
            yield "".join(batch)


def _count_chunks(path: str, lines_per_chunk: int):
    with open(path, "r", encoding="utf-8") as f:
        num_lines = sum(1 for _ in f)
    return (num_lines + lines_per_chunk - 1) // lines_per_chunk


def encode_file_to_uint16_parallel(
    vocab_path: str,
    merges_path: str,
    input_path: str,
    output_path: str,
    num_workers: int | None = None,
    lines_per_chunk: int = 20000,
):
    if num_workers is None:
        num_workers = os.cpu_count() or 1

    total_chunks = _count_chunks(input_path, lines_per_chunk)
    chunks = _batched_lines(input_path, lines_per_chunk)

    ctx = mp.get_context("spawn")
    encoded_parts = []

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(vocab_path, merges_path),
    ) as ex:

        results = ex.map(_encode_chunk, chunks, chunksize=1)

        for arr in tqdm(results, total=total_chunks, desc=f"Encoding {input_path}"):
            encoded_parts.append(arr)

    if encoded_parts:
        all_tokens = np.concatenate(encoded_parts)
    else:
        all_tokens = np.empty(0, dtype=np.uint16)

    np.save(output_path, all_tokens)
    print(f"Saved {len(all_tokens)} tokens to {output_path}")


def main():
    TINY_VOCAB = "artifacts/tinystories_vocab.pkl"
    TINY_MERGES = "artifacts/tinystories_merges.pkl"

    encode_file_to_uint16_parallel(
        TINY_VOCAB,
        TINY_MERGES,
        "../data/TinyStoriesV2-GPT4-valid.txt",
        "tiny_valid.npy",
        num_workers=10,
    )


if __name__ == "__main__":
    main()