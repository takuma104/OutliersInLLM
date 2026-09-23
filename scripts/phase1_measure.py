"""Phase 1 Day 2: measure M1-M8 for one model on the shared C4 probe set (plan §4.1 A-D).

Outputs (results/phase1/<model>/):
  norms.parquet            M4  λ_eff per residual norm × dim
  weights.parquet          M8  per nn.Linear weight stats
  residual.parquet         M1-M3 per residual read / block output
  residual_dims.parquet    per-dim stats of every residual read / block output
  linears.parquet          M5/M6 per nn.Linear input
  linear_channels.parquet  per-channel absmax / std of every nn.Linear input
  attention.parquet        M7 per full-attention layer × head (eager, subset of docs)
  tokens.npz               per-token M3 values (peak |u|, top-k energy, rms, category) per residual read
  lm.json                  PPL / BPB on WikiText-2 test and the C4 probe
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from outliers.data import (
    build_probe_set,
    c4_validation_texts,
    chunk_tokens,
    load_probe_doc_indices,
    token_byte_lengths,
    wikitext2_test_text,
)
from outliers.evaluate import evaluate_lm
from outliers.hooks import OutlierProbe
from outliers.models import load_model, set_attn_impl
from outliers.stats import norm_weight_table, weight_stats_table

WANDB_PROJECT = "OutliersInLLM"


def log(msg: str, t0: float) -> None:
    print(f"[{time.perf_counter() - t0:7.1f}s] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-docs", type=int, default=128)
    ap.add_argument("--attn-docs", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out-root", type=Path, default=Path("results/phase1"))
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    out = args.out_root / args.model
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(args.model)
    texts = c4_validation_texts()
    doc_idx = load_probe_doc_indices()[: args.n_docs]
    probe = build_probe_set(tok, texts, doc_idx, args.seq_len)
    log(f"loaded {args.model}, probe {tuple(probe.input_ids.shape)}", t0)

    norms = norm_weight_table(model)
    norms.to_parquet(out / "norms.parquet")
    weights = weight_stats_table(model)
    weights.to_parquet(out / "weights.parquet")
    log("static M4/M8 done", t0)

    with OutlierProbe(model) as p:
        p.run(probe.input_ids, probe.categories, batch_size=args.batch_size)
        res = p.results()
    res.residual.to_parquet(out / "residual.parquet")
    res.residual_dims.to_parquet(out / "residual_dims.parquet")
    res.linears.to_parquet(out / "linears.parquet")
    res.linear_channels.to_parquet(out / "linear_channels.parquet")
    np.savez_compressed(out / "tokens.npz", **res.tokens)
    del res
    log("main probe pass (M1-M3, M5, M6) done", t0)

    set_attn_impl(model, "eager")
    with OutlierProbe(model, residual=False, blocks=False, linears=False, attention=True) as p:
        n = min(args.attn_docs, probe.input_ids.shape[0])
        p.run(probe.input_ids[:n], probe.categories[:n], batch_size=1)
        attn = p.results().attention
    set_attn_impl(model, "sdpa")
    attn.to_parquet(out / "attention.parquet")
    log("attention pass (M7) done", t0)

    bl = token_byte_lengths(tok)
    lm = {
        "wikitext2": evaluate_lm(model, chunk_tokens(wikitext2_test_text(), tok, args.seq_len), bl),
        "c4_probe": evaluate_lm(model, probe.input_ids, bl),
    }
    (out / "lm.json").write_text(json.dumps(lm, indent=2))
    log(f"LM eval done: {json.dumps(lm)}", t0)

    if not args.no_wandb:
        log_wandb(args, out, lm)
    log("finished", t0)


def log_wandb(args: argparse.Namespace, out: Path, lm: dict[str, dict[str, float]]) -> None:
    import pandas as pd

    residual = pd.read_parquet(out / "residual.parquet")
    linears = pd.read_parquet(out / "linears.parquet")
    attn = pd.read_parquet(out / "attention.parquet")
    run = wandb.init(
        project=WANDB_PROJECT,
        group="phase1",
        job_type="measure",
        name=f"phase1-measure-{args.model}",
        config=vars(args) | {"model_key": args.model},
    )
    res = residual[residual.kind == "residual"]
    run.summary.update(
        {
            "wikitext2_ppl": lm["wikitext2"]["ppl"],
            "wikitext2_bpb": lm["wikitext2"]["bpb"],
            "c4_probe_ppl": lm["c4_probe"]["ppl"],
            "c4_probe_bpb": lm["c4_probe"]["bpb"],
            "residual_max_first": float(res.max_first.max()),
            "residual_max_delim": float(res.max_delim.max()),
            "residual_max_other": float(res.max_other.max()),
            "sink_score_max": float(res.sink_score.max()),
            "sink_score_median": float(res.sink_score.median()),
            "peak_p50_max": float(res.peak_p50.max()),
            "peak_p99_max": float(res.peak_p99.max()),
            "sink_ratio": float(attn.is_sink_head.mean()) if len(attn) else float("nan"),
            "down_proj_absmax_max": float(linears[linears.kind == "down_proj"].absmax.max()),
        }
    )
    for name, df in (("residual", residual), ("linears", linears), ("attention", attn)):
        run.log({name: wandb.Table(dataframe=df.astype({c: str for c in df.columns if df[c].dtype == object}))})
    art = wandb.Artifact(f"phase1-{args.model.replace('.', '_')}", type="phase1-stats")
    for f in sorted(out.glob("*.parquet")) + [out / "lm.json"]:
        art.add_file(str(f))
    run.log_artifact(art)
    run.finish()


if __name__ == "__main__":
    main()
