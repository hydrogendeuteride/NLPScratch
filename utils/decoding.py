from __future__ import annotations

from typing import List, Mapping, Optional, Sequence

import numpy as np


def to_cpu_np(x, np_backend=None) -> np.ndarray:
    """Best-effort backend -> numpy conversion (supports NumPy/CuPy)."""
    if np_backend is not None and hasattr(np_backend, "asnumpy"):
        try:
            return np_backend.asnumpy(x)
        except Exception:
            pass
    # CuPy ndarray also supports .get()
    if hasattr(x, "get"):
        try:
            return x.get()
        except Exception:
            pass
    return np.asarray(x)


def sample_top_k_top_p_from_logits(
    logits_1d: np.ndarray,
    *,
    topk: int = 0,
    topp: float = 1.0,
    temperature: float = 1.0,
    pad_id: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Sample a token id from logits using top-k and/or top-p (nucleus) filtering."""
    logits = np.asarray(logits_1d, dtype=np.float64)
    vocab_size = int(logits.shape[0])
    if vocab_size <= 0:
        return 0

    if temperature is None or float(temperature) <= 0:
        temperature = 1.0

    logits = logits / float(temperature)
    logits = logits - float(np.max(logits))
    probs = np.exp(logits)
    denom = float(np.sum(probs))
    if not np.isfinite(denom) or denom <= 0:
        return int(np.argmax(logits))
    probs = probs / denom

    if pad_id is not None and 0 <= int(pad_id) < vocab_size:
        probs[int(pad_id)] = 0.0

    topk = int(topk) if topk is not None else 0
    if topk > 0 and topk < vocab_size:
        top_idx = np.argpartition(probs, -topk)[-topk:]
        mask = np.zeros_like(probs, dtype=bool)
        mask[top_idx] = True
        probs = np.where(mask, probs, 0.0)

    # Renormalize before nucleus filtering so topp is measured in the filtered space.
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0:
        return int(np.argmax(logits))
    probs = probs / total

    if topp is not None and float(topp) < 1.0:
        topp = float(topp)
        if topp <= 0.0:
            return int(np.argmax(probs))

        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        cum = np.cumsum(sorted_probs)
        cutoff = int(np.searchsorted(cum, topp, side="left")) + 1
        cutoff = max(1, min(cutoff, vocab_size))

        nucleus_idx = sorted_idx[:cutoff]
        mask = np.zeros_like(probs, dtype=bool)
        mask[nucleus_idx] = True
        probs = np.where(mask, probs, 0.0)

        total = float(np.sum(probs))
        if not np.isfinite(total) or total <= 0:
            return int(np.argmax(logits))
        probs = probs / total

    choice = (rng.choice if rng is not None else np.random.choice)
    return int(choice(vocab_size, p=probs))


def generate_transformer_ids(
    model,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int,
    topk: int = 0,
    topp: float = 1.0,
    temperature: float = 1.0,
    stop_id: Optional[int] = None,
    pad_id: int = 0,
    seed: Optional[int] = None,
) -> List[int]:
    """Autoregressively generate ids from a numpy/cupy Transformer model."""
    npb = model.np
    out = [int(x) for x in prompt_ids]
    max_new_tokens = int(max_new_tokens)
    if max_new_tokens <= 0:
        return out

    rng = np.random.default_rng(int(seed)) if seed is not None else None

    for _ in range(max_new_tokens):
        seq = npb.array(out[-model.max_len :], dtype=npb.int32)[None, :]
        H, _ = model.forward(seq, return_hidden_only=True)
        next_logits = H[0, -1] @ model.W_vocab + model.b_vocab
        next_logits_cpu = to_cpu_np(next_logits, npb)
        next_id = sample_top_k_top_p_from_logits(
            next_logits_cpu,
            topk=topk,
            topp=topp,
            temperature=temperature,
            pad_id=pad_id,
            rng=rng,
        )
        out.append(int(next_id))
        if stop_id is not None and int(next_id) == int(stop_id):
            break

    return out


def ids_to_text(ids: Sequence[int], idx_to_token: Mapping[int, str]) -> str:
    tokens = [idx_to_token.get(int(i), "<UNK>") for i in ids]
    # Cheap BPE detok: "he@@ llo" -> "hello"
    return " ".join(tokens).replace("@@ ", "")
