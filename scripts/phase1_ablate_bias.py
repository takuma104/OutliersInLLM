"""Phase 1 §4.1 E follow-up: is the sink's direct contribution a constant (bias-like) signal?

With E[y_d] = mean of the norm output at the sink dim over non-first probe tokens (C4 calibration):
  direct_const : y_d <- E[y_d]                     (keep the denominator; direct part replaced by a constant)
  remove_const : x_d <- 0 inside the norm, y_d <- E[y_d]
                 (sink fully removed from the norm; its direct part folded into a constant = a Linear bias)
Both are applied at every hidden-size RMSNorm at once and evaluated on WikiText-2.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from outliers.data import (
    build_probe_set,
    c4_validation_texts,
    chunk_tokens,
    load_probe_doc_indices,
    token_byte_lengths,
    wikitext2_test_text,
)
from outliers.evaluate import evaluate_lm
from outliers.models import iter_residual_norms, load_model


@torch.no_grad()
def calibrate_output_means(
    model: nn.Module, targets: dict[str, tuple[nn.Module, int]], ids: torch.Tensor, batch_size: int = 4
) -> dict[str, float]:
    sums = {k: 0.0 for k in targets}
    n = 0
    handles = []
    for key, (mod, d) in targets.items():

        def hook(_m: nn.Module, _a: tuple, y: torch.Tensor, key: str = key, d: int = d) -> None:
            sums[key] += float(y[:, 1:, d].float().sum())

        handles.append(mod.register_forward_hook(hook))
    try:
        for s in range(0, ids.shape[0], batch_size):
            x = ids[s : s + batch_size].cuda()
            model.model(input_ids=x, use_cache=False)
            n += x.shape[0] * (x.shape[1] - 1)
    finally:
        for h in handles:
            h.remove()
    return {k: v / n for k, v in sums.items()}


@contextlib.contextmanager
def intervene(targets: dict[str, tuple[nn.Module, int]], means: dict[str, float], mode: str) -> Iterator[None]:
    handles = []
    for key, (mod, d) in targets.items():
        c = means[key]
        if mode == "remove_const":

            def pre(_m: nn.Module, args: tuple, d: int = d) -> tuple:
                x = args[0].clone()
                x[:, 1:, d] = 0
                return (x,) + args[1:]

            handles.append(mod.register_forward_pre_hook(pre))

        def post(_m: nn.Module, _a: tuple, y: torch.Tensor, d: int = d, c: float = c) -> torch.Tensor:
            y = y.clone()
            y[:, 1:, d] = c
            return y

        handles.append(mod.register_forward_hook(post))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", type=Path, default=Path("results/phase1"))
    ap.add_argument("--calib-docs", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    t0 = time.perf_counter()
    model, tok = load_model(args.model)
    residual = pd.read_parquet(args.root / args.model / "residual.parquet")
    sink = residual[residual.kind == "residual"].set_index("name")["sink_dim"]
    targets = {ni.name: (ni.module, int(sink[ni.name])) for ni in iter_residual_norms(model)}
    texts = c4_validation_texts()
    calib = build_probe_set(tok, texts, load_probe_doc_indices()[: args.calib_docs], args.seq_len).input_ids
    means = calibrate_output_means(model, targets, calib)
    windows = chunk_tokens(wikitext2_test_text(), tok, args.seq_len)
    bl = token_byte_lengths(tok)

    rows = [{"mode": "none"} | evaluate_lm(model, windows, bl, args.batch_size)]
    for mode in ("direct_const", "remove_const"):
        with intervene(targets, means, mode):
            rows.append({"mode": mode} | evaluate_lm(model, windows, bl, args.batch_size))
        print(f"[{time.perf_counter() - t0:5.0f}s] {mode} ppl={rows[-1]['ppl']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df["dppl_rel"] = df.ppl / df.ppl.iloc[0] - 1
    df.to_parquet(args.root / args.model / "ablate_bias.parquet")
    print(df[["mode", "ppl", "dppl_rel"]].to_string())


if __name__ == "__main__":
    main()
