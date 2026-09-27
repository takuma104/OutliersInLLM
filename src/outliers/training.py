"""Shared pieces of the distillation training loops (Phase 2 retrofit, Phase 3 recovery)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch


class TokenStream:
    """Packed token stream cut into ``seq_len`` sequences, visited in a fixed seeded order."""

    def __init__(self, path: Path, seq_len: int, seed: int) -> None:
        self.tokens = np.load(path, mmap_mode="r")
        self.seq_len = seq_len
        self.n_seq = len(self.tokens) // seq_len
        self.order = np.random.default_rng(seed).permutation(self.n_seq)

    def batch(self, start: int, count: int) -> torch.Tensor:
        rows = [self.order[(start + i) % self.n_seq] for i in range(count)]
        arr = np.stack([self.tokens[r * self.seq_len : (r + 1) * self.seq_len] for r in rows]).astype(np.int64)
        return torch.from_numpy(arr)


def lr_factor(step: int, total: int, warmup: int, min_ratio: float) -> float:
    """Linear warmup, then cosine decay to ``min_ratio``."""
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
