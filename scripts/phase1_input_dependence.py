"""Phase 1 §3.2 (optional): is the residual sink input-independent across domains?

Runs the residual-stream probe on English (C4 probe), Japanese (Wikipedia ja) and code (Python sources from the
installed site-packages), 32 × 2048 tokens each, and compares sink dims / scores per residual read.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from outliers.data import (
    c4_validation_texts,
    delimiter_token_mask,
    load_probe_doc_indices,
    select_probe_docs,
    token_categories,
    tokenize_prefixes,
)
from outliers.hooks import OutlierProbe
from outliers.models import MODEL_IDS, load_model

JA_SHARD = "20231101.ja/train-00014-of-00015.parquet"


def japanese_texts() -> list[str]:
    ds = load_dataset("wikimedia/wikipedia", data_files={"train": JA_SHARD}, split="train")
    return list(ds["text"])


def code_texts(seed: int = 0) -> list[str]:
    root = Path(torch.__file__).resolve().parents[1]  # site-packages
    files = sorted(p for p in root.glob("*/**/*.py") if 20_000 < p.stat().st_size < 400_000)
    random.Random(seed).shuffle(files)
    return [p.read_text(errors="ignore") for p in files[:2000]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--root", type=Path, default=Path("results/phase1"))
    args = ap.parse_args()

    models = ["qwen3.5-0.8b", "qwen3-0.6b"]
    toks: list[PreTrainedTokenizerBase] = [AutoTokenizer.from_pretrained(MODEL_IDS[m]) for m in models]
    c4 = c4_validation_texts()
    domains: dict[str, tuple[list[str], list[int]]] = {"en": (c4, load_probe_doc_indices()[: args.n_docs])}
    for name, texts in (("ja", japanese_texts()), ("code", code_texts())):
        domains[name] = (texts, select_probe_docs(texts, toks, args.n_docs, args.seq_len, min_chars=2048))

    rows = []
    for m, tok in zip(models, toks):
        model, _ = load_model(m)
        delim = delimiter_token_mask(tok)
        for dom, (texts, idx) in domains.items():
            ids = tokenize_prefixes(tok, texts, idx, args.seq_len)
            with OutlierProbe(model, blocks=False, linears=False) as p:
                p.run(ids, token_categories(ids, delim), batch_size=4)
                res = p.results().residual
            rows.append(res.assign(model=m, domain=dom))
            print(m, dom, "sink dims:", res.sink_dim.value_counts().head(3).to_dict(),
                  "score median", round(float(res.sink_score.median()), 1), flush=True)
        del model
        torch.cuda.empty_cache()
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(args.root / "input_dependence.parquet")
    summary = df[df["sub"] != "final"].groupby(["model", "domain"]).agg(
        sink_dim_mode=("sink_dim", lambda s: int(s.mode().iloc[0])),
        sink_dim_share=("sink_dim", lambda s: float((s == s.mode().iloc[0]).mean())),
        sink_score_median=("sink_score", "median"),
        argmax_hit_median=("sink_argmax_frac", "median"),
        peak_p50_median=("peak_p50", "median"),
        max_first=("max_first", "max"),
        max_other=("max_other", "max"),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
