from __future__ import annotations

from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from typing import BinaryIO, DefaultDict, Dict, List, Tuple
import os
import re

import regex as reg

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


GPT2_PRETOKENIZE_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"""
    r""" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

Word = Tuple[int, ...]
Pair = Tuple[int, int]


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> List[int]:
    
    assert isinstance(split_special_token, bytes), "split_special_token must be bytes"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desired_num_chunks <= 1 or file_size == 0:
        return [0, file_size]

    chunk_size = max(1, file_size // desired_num_chunks)
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


def _count_chunk_fast_eot(
    raw_chunk: bytes,
    eot_token: bytes,
) -> Counter[Word]:

    counter: Counter[Word] = Counter()

    for doc in raw_chunk.split(eot_token):
        if not doc:
            continue
        text = doc.decode("utf-8", errors="ignore")
        for match in reg.finditer(GPT2_PRETOKENIZE_PATTERN, text):
            piece = match.group(0).encode("utf-8")
            counter[tuple(piece)] += 1

    return counter


def _count_chunk_generic(
    raw_chunk: bytes,
    special_tokens: List[str],
) -> Counter[Word]:

    counter: Counter[Word] = Counter()
    text = raw_chunk.decode("utf-8", errors="ignore")
    parts = _split_on_special_tokens(text, special_tokens)
    special_token_set = set(special_tokens)

    for part in parts:
        if part in special_token_set:
            continue
        for match in reg.finditer(GPT2_PRETOKENIZE_PATTERN, part):
            piece = match.group(0).encode("utf-8")
            counter[tuple(piece)] += 1

    return counter


def _count_pretokens_in_file_chunk(
    args: Tuple[str, int, int, List[str]],
) -> Counter[Word]:
    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as f:
        f.seek(start)
        raw_chunk = f.read(end - start)

    if len(special_tokens) == 1:
        return _count_chunk_fast_eot(raw_chunk, special_tokens[0].encode("utf-8"))

    return _count_chunk_generic(raw_chunk, special_tokens)


def _build_pretoken_counts_parallel(
    input_path: str,
    special_tokens: List[str],
    num_processes: int | None = None,
    num_chunks: int | None = None,
    show_progress: bool = False,
) -> Counter[Word]:
    if num_processes is None:
        num_processes = max(1, min(4, cpu_count() - 1))

    if num_chunks is None:
        num_chunks = max(num_processes * 4, num_processes)

    with open(input_path, "rb") as f:
        if special_tokens:
            split_tok = special_tokens[0].encode("utf-8")
            boundaries = find_chunk_boundaries(f, num_chunks, split_tok)
        else:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(0)
            if num_chunks <= 1 or file_size == 0:
                boundaries = [0, file_size]
            else:
                chunk_size = max(1, file_size // num_chunks)
                boundaries = list(range(0, file_size, chunk_size))
                if boundaries[-1] != file_size:
                    boundaries.append(file_size)

    tasks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end > start
    ]

    if len(tasks) == 1:
        return _count_pretokens_in_file_chunk(tasks[0])

    total: Counter[Word] = Counter()
    with Pool(processes=num_processes) as pool:
        iterator = pool.imap_unordered(_count_pretokens_in_file_chunk, tasks, chunksize=1)
        if show_progress and tqdm is not None:
            iterator = tqdm(iterator, total=len(tasks), desc="Pretoken chunks", unit="chunk")
        for c in iterator:
            total.update(c)

    return total


def _word_pair_counter(word: Word) -> Counter[Pair]:
    c: Counter[Pair] = Counter()
    for i in range(len(word) - 1):
        c[(word[i], word[i + 1])] += 1
    return c


def _merge_word(word: Word, pair: Pair, merged_token_id: int) -> Word:
    a, b = pair
    new_word: List[int] = []

    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            new_word.append(merged_token_id)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def _build_pair_counts_and_index_by_id(
    word_freqs: Counter[Word],
) -> tuple[
    dict[int, Word],
    dict[int, int],
    Counter[Pair],
    DefaultDict[Pair, list[int]],
    int,
]:
    id_to_word: dict[int, Word] = {}
    word_freqs_by_id: dict[int, int] = {}
    pair_counts: Counter[Pair] = Counter()
    pair_to_word_ids: DefaultDict[Pair, list[int]] = defaultdict(list)

    next_word_id = 0
    for word, freq in word_freqs.items():
        wid = next_word_id
        next_word_id += 1

        id_to_word[wid] = word
        word_freqs_by_id[wid] = freq

        if len(word) < 2:
            continue

        local_pairs = _word_pair_counter(word)
        for pair, c in local_pairs.items():
            pair_counts[pair] += c * freq
            pair_to_word_ids[pair].append(wid)

    return id_to_word, word_freqs_by_id, pair_counts, pair_to_word_ids, next_word_id


def _word_contains_pair(word: Word, pair: Pair) -> bool:
    a, b = pair
    for i in range(len(word) - 1):
        if word[i] == a and word[i + 1] == b:
            return True
    return False


def _choose_best_pair(
    pair_counts: Counter[Pair],
    token_bytes: Dict[int, bytes],
) -> Pair | None:

    if not pair_counts:
        return None

    return max(
        pair_counts.items(),
        key=lambda kv: (
            kv[1],
            token_bytes[kv[0][0]],
            token_bytes[kv[0][1]],
        ),
    )[0]


def _incremental_apply_merge_by_id(
    id_to_word: dict[int, Word],
    word_freqs_by_id: dict[int, int],
    pair_counts: Counter[Pair],
    pair_to_word_ids: DefaultDict[Pair, list[int]],
    merge_pair: Pair,
    merged_token_id: int,
    next_word_id: int,
) -> int:
    candidate_ids = pair_to_word_ids.get(merge_pair, [])
    if not candidate_ids:
        pair_counts.pop(merge_pair, None)
        return next_word_id

    seen_ids = set()
    affected_ids: list[int] = []
    for wid in candidate_ids:
        if wid in seen_ids:
            continue
        seen_ids.add(wid)

        freq = word_freqs_by_id.get(wid, 0)
        if freq <= 0:
            continue

        old_word = id_to_word[wid]
        if _word_contains_pair(old_word, merge_pair):
            affected_ids.append(wid)

    if not affected_ids:
        pair_counts.pop(merge_pair, None)
        pair_to_word_ids.pop(merge_pair, None)
        return next_word_id

    for old_wid in affected_ids:
        freq = word_freqs_by_id.get(old_wid, 0)
        if freq <= 0:
            continue

        old_word = id_to_word[old_wid]
        new_word = _merge_word(old_word, merge_pair, merged_token_id)
        if new_word == old_word:
            continue

        old_local = _word_pair_counter(old_word)
        new_local = _word_pair_counter(new_word)

        for pair, c in old_local.items():
            pair_counts[pair] -= c * freq
            if pair_counts[pair] <= 0:
                pair_counts.pop(pair, None)

        word_freqs_by_id[old_wid] = 0

        new_wid = next_word_id
        next_word_id += 1

        id_to_word[new_wid] = new_word
        word_freqs_by_id[new_wid] = freq

        for pair, c in new_local.items():
            pair_counts[pair] += c * freq
            pair_to_word_ids[pair].append(new_wid)

    pair_to_word_ids.pop(merge_pair, None)
    return next_word_id


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    show_progress: bool = False,
    num_processes: int | None = None,
    num_chunks: int | None = None,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_token_id = 256

    for tok in special_tokens:
        if next_token_id >= vocab_size:
            return vocab, []
        vocab[next_token_id] = tok.encode("utf-8")
        next_token_id += 1

    if show_progress:
        print("[train_bpe] Building pretoken counts...")
    word_freqs = _build_pretoken_counts_parallel(
        input_path=input_path,
        special_tokens=special_tokens,
        num_processes=num_processes,
        num_chunks=num_chunks,
        show_progress=show_progress,
    )
    if show_progress:
        print(f"[train_bpe] Done pretokenization: {len(word_freqs)} unique pretokens")

    id_to_word, word_freqs_by_id, pair_counts, pair_to_word_ids, next_word_id = (
        _build_pair_counts_and_index_by_id(word_freqs)
    )

    merges: List[Tuple[bytes, bytes]] = []

    target_merges = vocab_size - next_token_id
    iterator = range(target_merges)
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, desc="BPE merges", unit="merge")

    for _ in iterator:
        best_pair = _choose_best_pair(pair_counts, vocab)
        if best_pair is None:
            break

        merged_token_bytes = vocab[best_pair[0]] + vocab[best_pair[1]]
        merged_token_id = next_token_id

        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        vocab[merged_token_id] = merged_token_bytes
        next_token_id += 1

        next_word_id = _incremental_apply_merge_by_id(
            id_to_word=id_to_word,
            word_freqs_by_id=word_freqs_by_id,
            pair_counts=pair_counts,
            pair_to_word_ids=pair_to_word_ids,
            merge_pair=best_pair,
            merged_token_id=merged_token_id,
            next_word_id=next_word_id,
        )

    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe(
        input_path="../data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        show_progress=True,
        num_processes=3,
        num_chunks=12,
    )
    print(f"Learned vocab size: {len(vocab)}")
    print(f"Num merges: {len(merges)}")
