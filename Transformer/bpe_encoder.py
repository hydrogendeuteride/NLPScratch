"""Byte Pair Encoding (BPE) tokenizer/encoder for the Transformer code.

This implementation is intentionally lightweight (no external deps) and focuses on:
- Training BPE merges from a token corpus
- Encoding word tokens into BPE subword tokens
- Mapping subword tokens to ids and producing PyTorch tensors

Typical usage (word-level corpus):

    from Transformer.bpe_encoder import BPEEncoder
    bpe = BPEEncoder()
    bpe.train(token_sequences, vocab_size=8000)
    ids = bpe.encode_ids(["This", "is", "a", "test"])
    x = bpe.encode_to_padded_tensor([["This", "is"], ["Another", "test"]])
    bpe.save("weight/bpe.pt")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch


Pair = Tuple[str, str]


@dataclass(frozen=True)
class BPESpecialTokens:
    pad: str = "<PAD>"
    unk: str = "<UNK>"
    bos: str = "<BOS>"
    eos: str = "<EOS>"


class BPEEncoder:
    def __init__(
        self,
        *,
        special: BPESpecialTokens = BPESpecialTokens(),
        end_of_word: str = "</w>",
        continuation: str = "@@",
        lowercase: bool = False,
    ) -> None:
        self.special = special
        self.end_of_word = str(end_of_word)
        self.continuation = str(continuation)
        self.lowercase = bool(lowercase)

        self.merges: List[Pair] = []
        self.bpe_ranks: Dict[Pair, int] = {}
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._cache: Dict[str, List[str]] = {}

    # -------------------------
    # Training
    # -------------------------
    def train(
        self,
        token_sequences: Iterable[Sequence[str]],
        *,
        vocab_size: int,
        min_frequency: int = 2,
        max_word_types: Optional[int] = None,
    ) -> "BPEEncoder":
        """Learn merges from a corpus of word tokens.

        Args:
            token_sequences: iterable of token lists (word-level tokens).
            vocab_size: target size of subword vocabulary (including special tokens).
            min_frequency: minimum pair frequency to merge.
            max_word_types: optionally cap number of unique words considered (top by frequency).
        """
        if vocab_size < 8:
            raise ValueError("vocab_size is too small for BPE + special tokens.")
        if min_frequency < 1:
            raise ValueError("min_frequency must be >= 1")

        word_freq: Dict[str, int] = {}
        for seq in token_sequences:
            for w in seq:
                if self.lowercase:
                    w = w.lower()
                if not w:
                    continue
                word_freq[w] = word_freq.get(w, 0) + 1

        if not word_freq:
            raise ValueError("Empty corpus: no tokens to train on.")

        if max_word_types is not None and max_word_types > 0 and len(word_freq) > max_word_types:
            # Keep only most frequent word types to speed up training.
            word_freq = dict(
                sorted(word_freq.items(), key=lambda kv: (-kv[1], kv[0]))[: int(max_word_types)]
            )

        # Represent each word as a tuple of symbols (chars) + EOW marker.
        word_syms: Dict[str, Tuple[str, ...]] = {
            w: tuple(list(w) + [self.end_of_word]) for w in word_freq.keys()
        }

        # Greedy merges until vocab_size reached (approx) or no pair qualifies.
        self.merges = []
        self._cache.clear()

        def current_symbol_vocab_size() -> int:
            symbols = set()
            for syms in word_syms.values():
                symbols.update(syms)
            return len(symbols)

        # Ensure special tokens are accounted for.
        target_subword_vocab = max(0, int(vocab_size) - len(self._special_token_list()))

        while current_symbol_vocab_size() < target_subword_vocab:
            pair_freqs = self._get_pair_stats(word_syms, word_freq)
            if not pair_freqs:
                break

            best_pair, best_freq = max(pair_freqs.items(), key=lambda kv: (kv[1], kv[0]))
            if best_freq < min_frequency:
                break

            self.merges.append(best_pair)
            word_syms = self._merge_pair_in_vocab(best_pair, word_syms)

        # Build ranks for fast encoding.
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}

        # Build a subword vocabulary from the training corpus under learned merges.
        subword_freq: Dict[str, int] = {}
        for w, freq in word_freq.items():
            for s in self.encode_word(w):
                subword_freq[s] = subword_freq.get(s, 0) + freq

        # Reserve ids: PAD=0, UNK=1, BOS=2, EOS=3 (stable, Transformer code uses PAD=0).
        tokens_sorted = sorted(subword_freq.items(), key=lambda kv: (-kv[1], kv[0]))
        subwords = [t for t, _ in tokens_sorted]
        self._set_vocab_from_tokens(subwords[:target_subword_vocab])

        return self

    @staticmethod
    def _get_pair_stats(
        word_syms: Mapping[str, Tuple[str, ...]],
        word_freq: Mapping[str, int],
    ) -> Dict[Pair, int]:
        pair_freqs: Dict[Pair, int] = {}
        for w, syms in word_syms.items():
            freq = int(word_freq.get(w, 0))
            if freq <= 0 or len(syms) < 2:
                continue
            prev = syms[0]
            for cur in syms[1:]:
                pair = (prev, cur)
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
                prev = cur
        return pair_freqs

    @staticmethod
    def _merge_pair_in_vocab(
        pair: Pair, word_syms: Mapping[str, Tuple[str, ...]]
    ) -> Dict[str, Tuple[str, ...]]:
        a, b = pair
        merged = a + b
        out: Dict[str, Tuple[str, ...]] = {}
        for w, syms in word_syms.items():
            if len(syms) < 2:
                out[w] = syms
                continue
            new_syms: List[str] = []
            i = 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                    new_syms.append(merged)
                    i += 2
                else:
                    new_syms.append(syms[i])
                    i += 1
            out[w] = tuple(new_syms)
        return out

    # -------------------------
    # Encoding
    # -------------------------
    def encode_word(self, word: str) -> List[str]:
        """Encode a single word into BPE subword tokens."""
        if self.lowercase:
            word = word.lower()

        cached = self._cache.get(word)
        if cached is not None:
            return list(cached)

        if not word:
            self._cache[word] = []
            return []

        # Start from character symbols + EOW.
        syms = tuple(list(word) + [self.end_of_word])

        if not self.bpe_ranks:
            # Untrained: fall back to whole-word as a single token.
            out = [word]
            self._cache[word] = out
            return list(out)

        while True:
            pairs = self._adjacent_pairs(syms)
            if not pairs:
                break

            # pick lowest-rank pair (earliest merge)
            best = min(pairs, key=lambda p: self.bpe_ranks.get(p, 1_000_000_000))
            if best not in self.bpe_ranks:
                break
            syms = tuple(self._merge_pair_in_symbols(best, syms))

        out = self._finalize_symbols(syms)
        self._cache[word] = out
        return list(out)

    def encode_tokens(self, tokens: Sequence[str]) -> List[str]:
        """Encode a list of word tokens into BPE subword tokens."""
        out: List[str] = []
        for w in tokens:
            out.extend(self.encode_word(w))
        return out

    def encode_ids(
        self,
        tokens: Sequence[str],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode tokens and map to ids using this encoder's vocabulary."""
        subwords = self.encode_tokens(tokens)
        ids: List[int] = []
        if add_bos:
            ids.append(self.token_to_id.get(self.special.bos, 2))
        unk_id = self.token_to_id.get(self.special.unk, 1)
        for t in subwords:
            ids.append(self.token_to_id.get(t, unk_id))
        if add_eos:
            ids.append(self.token_to_id.get(self.special.eos, 3))
        return ids

    def encode_to_tensor(
        self,
        tokens: Sequence[str],
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.long,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> torch.Tensor:
        return torch.tensor(
            self.encode_ids(tokens, add_bos=add_bos, add_eos=add_eos),
            dtype=dtype,
            device=device,
        )

    def encode_to_padded_tensor(
        self,
        batch_tokens: Sequence[Sequence[str]],
        *,
        max_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> torch.Tensor:
        """Encode a batch into a padded LongTensor shaped (B, S)."""
        encoded = [self.encode_ids(t, add_bos=add_bos, add_eos=add_eos) for t in batch_tokens]
        if max_len is None:
            max_len = max((len(x) for x in encoded), default=0)
        pad_id = self.token_to_id.get(self.special.pad, 0)
        out = torch.full((len(encoded), int(max_len)), pad_id, dtype=torch.long, device=device)
        for i, ids in enumerate(encoded):
            if not ids:
                continue
            ids = ids[: int(max_len)]
            out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        return out

    @staticmethod
    def _adjacent_pairs(syms: Tuple[str, ...]) -> List[Pair]:
        if len(syms) < 2:
            return []
        return [(syms[i], syms[i + 1]) for i in range(len(syms) - 1)]

    @staticmethod
    def _merge_pair_in_symbols(pair: Pair, syms: Tuple[str, ...]) -> List[str]:
        a, b = pair
        merged = a + b
        out: List[str] = []
        i = 0
        while i < len(syms):
            if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(syms[i])
                i += 1
        return out

    def _finalize_symbols(self, syms: Tuple[str, ...]) -> List[str]:
        # Convert internal symbols with EOW marker into display tokens using continuation marks.
        if not syms:
            return []

        clean: List[str] = []
        last_idx: Optional[int] = None

        for s in syms:
            if s == self.end_of_word:
                if clean:
                    last_idx = len(clean) - 1
                break
            if s.endswith(self.end_of_word):
                s = s[: -len(self.end_of_word)]
                if s:
                    clean.append(s)
                    last_idx = len(clean) - 1
                break
            clean.append(s)

        if not clean:
            return []

        if last_idx is None:
            last_idx = len(clean) - 1

        out: List[str] = []
        for i, s in enumerate(clean):
            if i == last_idx:
                out.append(s)
            else:
                out.append(s + self.continuation)
        return out

    # -------------------------
    # Vocab / persistence
    # -------------------------
    def _special_token_list(self) -> List[str]:
        return [self.special.pad, self.special.unk, self.special.bos, self.special.eos]

    def _set_vocab_from_tokens(self, tokens: Sequence[str]) -> None:
        token_to_id: Dict[str, int] = {}
        for i, s in enumerate(self._special_token_list()):
            token_to_id[s] = i

        next_id = len(token_to_id)
        for t in tokens:
            if t in token_to_id:
                continue
            token_to_id[t] = next_id
            next_id += 1

        self.token_to_id = token_to_id
        self.id_to_token = {i: t for t, i in token_to_id.items()}

    def save(self, path: str) -> None:
        """Save merges/vocab to a torch checkpoint."""
        state = {
            "special": {
                "pad": self.special.pad,
                "unk": self.special.unk,
                "bos": self.special.bos,
                "eos": self.special.eos,
            },
            "end_of_word": self.end_of_word,
            "continuation": self.continuation,
            "lowercase": self.lowercase,
            "merges": self.merges,
            "token_to_id": self.token_to_id,
        }
        torch.save(state, path)

    @classmethod
    def load(
        cls, path: str, *, map_location: Optional[Union[str, torch.device]] = "cpu"
    ) -> "BPEEncoder":
        state = torch.load(path, map_location=map_location)
        special = BPESpecialTokens(
            pad=state["special"]["pad"],
            unk=state["special"]["unk"],
            bos=state["special"]["bos"],
            eos=state["special"]["eos"],
        )
        enc = cls(
            special=special,
            end_of_word=state["end_of_word"],
            continuation=state["continuation"],
            lowercase=bool(state.get("lowercase", False)),
        )
        enc.merges = [tuple(x) for x in state.get("merges", [])]
        enc.bpe_ranks = {pair: i for i, pair in enumerate(enc.merges)}
        enc.token_to_id = {str(k): int(v) for k, v in state.get("token_to_id", {}).items()}
        enc.id_to_token = {i: t for t, i in enc.token_to_id.items()}
        return enc

    @property
    def pad_id(self) -> int:
        return int(self.token_to_id.get(self.special.pad, 0))

    @property
    def unk_id(self) -> int:
        return int(self.token_to_id.get(self.special.unk, 1))

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)


if __name__ == "__main__":
    # Minimal smoke test (no repo imports): train on a tiny toy corpus.
    corpus = [["low", "lower", "newest"], ["widest", "low"], ["low", "newer"]]
    bpe = BPEEncoder(lowercase=True).train(corpus, vocab_size=64, min_frequency=1)
    print("vocab_size:", bpe.vocab_size)
    print("merges:", len(bpe.merges))
    print("encode(lowest):", bpe.encode_tokens(["lowest"]))
    print("encode(newest):", bpe.encode_tokens(["newest"]))
    print("ids:", bpe.encode_ids(["lowest"]))
