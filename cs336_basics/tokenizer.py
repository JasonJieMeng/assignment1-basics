from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Tuple
import pickle
import re

import regex as reg


GPT2_PRETOKENIZE_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"""
    r""" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

Word = Tuple[bytes, ...]
Pair = Tuple[bytes, bytes]


class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] | None = None,
    ):
        self.vocab: Dict[int, bytes] = dict(vocab)
        self.merges: List[Tuple[bytes, bytes]] = list(merges)
        self.special_tokens: List[str] = list(special_tokens or [])

        self.token_to_id: Dict[bytes, int] = {token: idx for idx, token in self.vocab.items()}

        next_id = max(self.vocab.keys(), default=-1) + 1
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in self.token_to_id:
                self.vocab[next_id] = tok_bytes
                self.token_to_id[tok_bytes] = next_id
                next_id += 1

        self.special_token_bytes = {tok.encode("utf-8") for tok in self.special_tokens}
        self.special_token_set = set(self.special_tokens)

        # pair -> rank (smaller means earlier merge, therefore higher priority)
        self.merge_rank: Dict[Pair, int] = {
            pair: i for i, pair in enumerate(self.merges)
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: List[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> List[int]:
        return list(self.encode_iterable([text]))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self._encode_text(chunk)

    def decode(self, ids: List[int]) -> str:
        pieces: List[bytes] = []
        for idx in ids:
            if idx not in self.vocab:
                raise KeyError(f"Unknown token id: {idx}")
            pieces.append(self.vocab[idx])
        return b"".join(pieces).decode("utf-8", errors="replace")

    def _encode_text(self, text: str) -> Iterator[int]:
        for part, is_special in self._split_with_special_tokens(text):
            if is_special:
                tok_id = self.token_to_id[part.encode("utf-8")]
                yield tok_id
                continue

            for match in reg.finditer(GPT2_PRETOKENIZE_PATTERN, part):
                pretoken = match.group(0)
                yield from self._encode_pretoken(pretoken)

    def _split_with_special_tokens(self, text: str) -> List[Tuple[str, bool]]:
        if not self.special_tokens:
            return [(text, False)]

        # Longest-first so overlapping special tokens are handled correctly.
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(tok) for tok in sorted_special_tokens) + ")"
        parts = re.split(pattern, text)

        out: List[Tuple[str, bool]] = []
        for part in parts:
            if part == "":
                continue
            out.append((part, part in self.special_token_set))
        return out

    def _encode_pretoken(self, pretoken: str) -> List[int]:
        word: Word = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
        merged_word = self._apply_merges(word)
        return [self.token_to_id[token] for token in merged_word]

    def _apply_merges(self, word: Word) -> Word:
        # Apply merges in the same order of creation.
        # Equivalent greedy implementation: repeatedly merge the adjacent pair
        # with the smallest merge rank among currently present pairs.
        while len(word) >= 2:
            best_index = -1
            best_rank = None

            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_index = i

            if best_index == -1:
                break

            a = word[best_index]
            b = word[best_index + 1]
            merged = a + b
            word = word[:best_index] + (merged,) + word[best_index + 2 :]

        return word