"""LM quality: chunked NLL -> PPL / bits-per-byte (plan §2: BPB for cross-model, PPL for within-model)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


@torch.no_grad()
def token_nll(model: PreTrainedModel, input_ids: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """Per-token NLL [B, T-1] (fp32) of next-token prediction; lm_head is applied in sequence chunks."""
    h = model.model(input_ids=input_ids, use_cache=False).last_hidden_state[:, :-1]
    targets = input_ids[:, 1:]
    out = torch.empty(targets.shape, device=h.device, dtype=torch.float32)
    for s in range(0, h.shape[1], chunk):
        logits = model.lm_head(h[:, s : s + chunk]).float()
        out[:, s : s + chunk] = F.cross_entropy(
            logits.transpose(1, 2), targets[:, s : s + chunk], reduction="none"
        )
    return out


@torch.no_grad()
def evaluate_lm(
    model: PreTrainedModel,
    windows: torch.Tensor,
    byte_lengths: torch.Tensor,
    batch_size: int = 8,
) -> dict[str, float]:
    """PPL and BPB over non-overlapping windows [N, T] (the first token of each window is context only)."""
    dev = next(model.parameters()).device
    nll_sum, n_tok, n_bytes = 0.0, 0, 0
    for s in range(0, windows.shape[0], batch_size):
        ids = windows[s : s + batch_size].to(dev)
        nll = token_nll(model, ids)
        nll_sum += float(nll.double().sum())
        n_tok += nll.numel()
        n_bytes += int(byte_lengths.to(dev)[ids[:, 1:]].sum())
    return {
        "nll": nll_sum / n_tok,
        "ppl": math.exp(nll_sum / n_tok),
        "bpb": nll_sum / math.log(2) / n_bytes,
        "n_tokens": n_tok,
        "n_bytes": n_bytes,
    }
