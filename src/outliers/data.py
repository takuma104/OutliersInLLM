"""Probe / evaluation data (plan §3.2)."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

C4_VALIDATION_SHARD = "en/c4-validation.00000-of-00008.json.gz"
PROBE_DOCS_PATH = Path(__file__).resolve().parents[2] / "data" / "probe_c4_docs.json"

# token categories for M1 (plan §3.3); first token is always index 0 of a sequence
CAT_FIRST, CAT_DELIM, CAT_OTHER = 0, 1, 2
CATEGORY_NAMES: tuple[str, str, str] = ("first", "delim", "other")

_DELIM_CHARS = frozenset(".,;:!?")


def c4_validation_texts() -> list[str]:
    ds = load_dataset("allenai/c4", data_files={"validation": C4_VALIDATION_SHARD}, split="validation")
    return list(ds["text"])


def wikitext2_test_text() -> str:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(ds["text"])


def select_probe_docs(
    texts: Sequence[str],
    tokenizers: Sequence[PreTrainedTokenizerBase],
    n_docs: int = 128,
    seq_len: int = 2048,
    seed: int = 0,
    min_chars: int = 4096,
) -> list[int]:
    """Pick documents that are at least ``seq_len`` tokens long under every tokenizer.

    Sequences are cut from the document start (plan §3.2), so the same documents give comparable
    first-token behaviour across models.
    """
    order = [i for i, t in enumerate(texts) if len(t) >= min_chars]
    random.Random(seed).shuffle(order)
    chosen: list[int] = []
    for i in order:
        if all(len(tok(texts[i]).input_ids) >= seq_len for tok in tokenizers):
            chosen.append(i)
            if len(chosen) == n_docs:
                return chosen
    raise ValueError(f"only {len(chosen)} documents with >= {seq_len} tokens")


def load_probe_doc_indices(path: Path = PROBE_DOCS_PATH) -> list[int]:
    """Doc indices written by scripts/phase1_make_probe.py."""
    return list(json.loads(path.read_text())["doc_indices"])


def tokenize_prefixes(
    tokenizer: PreTrainedTokenizerBase, texts: Sequence[str], indices: Sequence[int], seq_len: int
) -> torch.Tensor:
    rows = [tokenizer(texts[i]).input_ids[:seq_len] for i in indices]
    if any(len(r) < seq_len for r in rows):
        raise ValueError("document shorter than seq_len")
    return torch.tensor(rows, dtype=torch.long)


def chunk_tokens(text: str, tokenizer: PreTrainedTokenizerBase, seq_len: int) -> torch.Tensor:
    """Non-overlapping ``seq_len`` windows of the tokenized text (the tail remainder is dropped)."""
    ids = tokenizer(text).input_ids
    n = len(ids) // seq_len
    return torch.tensor(ids[: n * seq_len], dtype=torch.long).view(n, seq_len)


def token_byte_lengths(tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """UTF-8 byte length of each vocab entry (0 for special tokens).

    Qwen tokenizers are byte-level BPE, where each character of the token string maps to exactly one byte.
    """
    n = len(tokenizer)
    special = set(tokenizer.all_special_ids) | set(tokenizer.added_tokens_decoder.keys())
    toks = tokenizer.convert_ids_to_tokens(list(range(n)))
    lengths = [0 if (i in special or t is None) else len(t) for i, t in enumerate(toks)]
    return torch.tensor(lengths, dtype=torch.long)


def delimiter_token_mask(tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """Bool mask over the vocab: tokens made only of whitespace / ``.,;:!?`` containing a period or newline."""
    n = len(tokenizer)
    mask = torch.zeros(n, dtype=torch.bool)
    for i, s in enumerate(tokenizer.batch_decode([[i] for i in range(n)])):
        if not s or "�" in s:
            continue
        if ("." in s or "\n" in s) and all(c.isspace() or c in _DELIM_CHARS for c in s):
            mask[i] = True
    return mask


def token_categories(input_ids: torch.Tensor, delim_mask: torch.Tensor) -> torch.Tensor:
    """[B, T] -> category ids (CAT_FIRST / CAT_DELIM / CAT_OTHER)."""
    cats = torch.full_like(input_ids, CAT_OTHER)
    cats[delim_mask.to(input_ids.device)[input_ids]] = CAT_DELIM
    cats[:, 0] = CAT_FIRST
    return cats


@dataclass
class ProbeSet:
    """Tokenized probe sequences for one model plus per-token metadata."""

    input_ids: torch.Tensor  # [N, T]
    categories: torch.Tensor  # [N, T]
    doc_indices: list[int]


def build_probe_set(
    tokenizer: PreTrainedTokenizerBase, texts: Sequence[str], doc_indices: Sequence[int], seq_len: int
) -> ProbeSet:
    ids = tokenize_prefixes(tokenizer, texts, doc_indices, seq_len)
    cats = token_categories(ids, delimiter_token_mask(tokenizer))
    return ProbeSet(input_ids=ids, categories=cats, doc_indices=list(doc_indices))
