# train_bpe.py

from __future__ import annotations

from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from typing import BinaryIO, DefaultDict, Dict, List, Set, Tuple
import os
import re

import regex as reg

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback if tqdm is unavailable
    tqdm = None


GPT2_PRETOKENIZE_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"""
    r""" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


Word = Tuple[bytes, ...]
Pair = Tuple[bytes, bytes]


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> List[int]:
    """
    Chunk the file into parts that can be processed independently.
    Boundary locations are shifted forward to the next occurrence of
    split_special_token, following the assignment starter idea.
    """
    assert isinstance(split_special_token, bytes), "split_special_token must be bytes"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desired_num_chunks <= 1 or file_size == 0:
        return [0, file_size]

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _split_on_special_tokens(text: str, special_tokens: List[str]) -> List[str]:
    if not special_tokens:
        return [text]
    pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
    return [part for part in re.split(pattern, text) if part != ""]


def _pretokenize_segment(segment: str) -> List[Word]:
    out: List[Word] = []
    for match in reg.finditer(GPT2_PRETOKENIZE_PATTERN, segment):
        piece = match.group(0).encode("utf-8")
        out.append(tuple(bytes([b]) for b in piece))
    return out


def _count_pretokens_in_chunk(args: Tuple[str, List[str]]) -> Counter[Word]:
    """
    Worker for multiprocessing:
    - split one chunk on special tokens
    - pretokenize non-special spans
    - return frequency counts of pretokens
    """
    chunk_text, special_tokens = args
    parts = _split_on_special_tokens(chunk_text, special_tokens)
    special_token_set = set(special_tokens)

    counter: Counter[Word] = Counter()
    for part in parts:
        if part in special_token_set:
            continue
        counter.update(_pretokenize_segment(part))
    return counter


def _build_pretoken_counts_parallel(
    input_path: str,
    special_tokens: List[str],
    num_processes: int | None = None,
) -> Counter[Word]:
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)

    with open(input_path, "rb") as f:
        if special_tokens:
            split_tok = special_tokens[0].encode("utf-8")
            boundaries = find_chunk_boundaries(f, num_processes, split_tok)
        else:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(0)
            if num_processes <= 1 or file_size == 0:
                boundaries = [0, file_size]
            else:
                chunk_size = max(1, file_size // num_processes)
                boundaries = list(range(0, file_size, chunk_size))
                if boundaries[-1] != file_size:
                    boundaries.append(file_size)

        chunk_texts: List[str] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_texts.append(chunk)

    if len(chunk_texts) == 1:
        return _count_pretokens_in_chunk((chunk_texts[0], special_tokens))

    with Pool(processes=num_processes) as pool:
        counters = pool.map(
            _count_pretokens_in_chunk,
            [(chunk_text, special_tokens) for chunk_text in chunk_texts],
        )

    total: Counter[Word] = Counter()
    for c in counters:
        total.update(c)
    return total


def _word_pair_counter(word: Word) -> Counter[Pair]:
    """
    Adjacent pair counts inside a single word, unweighted by frequency.
    """
    c: Counter[Pair] = Counter()
    for i in range(len(word) - 1):
        c[(word[i], word[i + 1])] += 1
    return c


def _merge_word(word: Word, pair: Pair) -> Word:
    """
    Merge every occurrence of `pair` in `word`.
    """
    a, b = pair
    merged = a + b
    new_word: List[bytes] = []

    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            new_word.append(merged)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def _build_pair_counts_and_index(
    word_freqs: Counter[Word],
) -> Tuple[Counter[Pair], DefaultDict[Pair, Set[Word]]]:
    """
    Build:
    - global pair counts
    - index mapping pair -> set of words that contain that pair
    """
    pair_counts: Counter[Pair] = Counter()
    pair_to_words: DefaultDict[Pair, Set[Word]] = defaultdict(set)

    for word, freq in word_freqs.items():
        if len(word) < 2:
            continue
        local_pairs = _word_pair_counter(word)
        for pair, c in local_pairs.items():
            pair_counts[pair] += c * freq
            pair_to_words[pair].add(word)

    return pair_counts, pair_to_words


def _choose_best_pair(pair_counts: Counter[Pair]) -> Pair | None:
    """
    Deterministic tie-breaking:
    max count, then lexicographically greatest pair.
    """
    if not pair_counts:
        return None

    return max(
        pair_counts.items(),
        key=lambda kv: (kv[1], kv[0][0], kv[0][1]),
    )[0]


def _incremental_apply_merge(
    word_freqs: Counter[Word],
    pair_counts: Counter[Pair],
    pair_to_words: DefaultDict[Pair, Set[Word]],
    merge_pair: Pair,
) -> None:
    """
    Incrementally update:
    - word_freqs
    - pair_counts
    - pair_to_words

    Only words containing `merge_pair` are touched.
    """
    affected_words = list(pair_to_words.get(merge_pair, set()))
    if not affected_words:
        # Defensive fallback.
        pair_counts.pop(merge_pair, None)
        return

    # We'll rebuild memberships for touched words locally.
    for old_word in affected_words:
        freq = word_freqs.get(old_word, 0)
        if freq == 0:
            continue

        new_word = _merge_word(old_word, merge_pair)
        if new_word == old_word:
            continue

        old_local = _word_pair_counter(old_word)
        new_local = _word_pair_counter(new_word)

        # Remove old weighted pair contributions and memberships.
        for pair, c in old_local.items():
            pair_counts[pair] -= c * freq
            if pair_counts[pair] <= 0:
                pair_counts.pop(pair, None)

            words_set = pair_to_words.get(pair)
            if words_set is not None:
                words_set.discard(old_word)
                if not words_set:
                    pair_to_words.pop(pair, None)

        # Update word frequencies.
        word_freqs[old_word] -= freq
        if word_freqs[old_word] == 0:
            del word_freqs[old_word]

        word_freqs[new_word] += freq

        # Add new weighted pair contributions and memberships.
        for pair, c in new_local.items():
            pair_counts[pair] += c * freq
            pair_to_words[pair].add(new_word)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    show_progress: bool = False,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Args:
        input_path: path to UTF-8 training text
        vocab_size: final vocabulary size cap, including:
            - 256 raw bytes
            - special tokens
            - merged tokens
        special_tokens: added to vocab; treated as hard boundaries during training
        show_progress: whether to show progress bars / phase logs during training

    Returns:
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    # Initial byte vocabulary.
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_token_id = 256

    # Add special tokens to vocab.
    for tok in special_tokens:
        if next_token_id >= vocab_size:
            return vocab, []
        vocab[next_token_id] = tok.encode("utf-8")
        next_token_id += 1

    # Parallel pre-tokenization / counting.
    if show_progress:
        print("[train_bpe] Building pretoken counts...")
    word_freqs = _build_pretoken_counts_parallel(
        input_path=input_path,
        special_tokens=special_tokens,
    )
    if show_progress:
        print(f"[train_bpe] Done pretokenization: {len(word_freqs)} unique pretokens")

    # Initial global pair counts and inverted index.
    pair_counts, pair_to_words = _build_pair_counts_and_index(word_freqs)

    merges: List[Pair] = []

    target_merges = vocab_size - next_token_id
    iterator = range(target_merges)
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, desc="BPE merges", unit="merge")

    for _ in iterator:
        best_pair = _choose_best_pair(pair_counts)
        if best_pair is None:
            break

        # Record merge and add merged token to vocab.
        merges.append(best_pair)
        vocab[next_token_id] = best_pair[0] + best_pair[1]
        next_token_id += 1

        # Incrementally update state.
        _incremental_apply_merge(
            word_freqs=word_freqs,
            pair_counts=pair_counts,
            pair_to_words=pair_to_words,
            merge_pair=best_pair,
        )

    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )