"""Phase 2 follow-up: per-kind quantization sensitivity (one Linear kind quantized at a time) for checkpoints.

Distinguishes "weights became harder to quantize" from "the model became more sensitive to the same quantization
error" (the weight statistics themselves are compared in eval/weights.parquet).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from outliers.data import chunk_tokens, token_byte_lengths, wikitext2_test_text
from outliers.evaluate import evaluate_lm
from outliers.models import load_model
from outliers.quant import SCHEMES, fake_quantize
from outliers.retrofit import load_retrofit

KINDS = ["qkv", "o_proj", "gate_up", "down_proj"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="original model key, evaluated first")
    ap.add_argument("--ckpts", type=Path, nargs="+", required=True)
    ap.add_argument("--schemes", nargs="+", default=["W4A16", "A4-INT"])
    ap.add_argument("--windows", type=int, default=24)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for label, path in [("original", None)] + [(p.parent.name, p) for p in args.ckpts]:
        model, tok = load_model(args.base) if path is None else load_retrofit(path)[:2]
        windows = chunk_tokens(wikitext2_test_text(), tok, 2048)[: args.windows]
        bl = token_byte_lengths(tok)
        base = evaluate_lm(model, windows, bl, batch_size=2)["ppl"]
        for scheme in args.schemes:
            for kind in KINDS:
                with fake_quantize(model, SCHEMES[scheme], kinds=[kind]):
                    ppl = evaluate_lm(model, windows, bl, batch_size=2)["ppl"]
                rows.append({"model": label, "scheme": scheme, "kind": kind, "dppl_rel": ppl / base - 1})
        print(label, "done", flush=True)
        del model
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(df.pivot_table(index=["scheme", "kind"], columns="model", values="dppl_rel", sort=False).round(4).to_string())


if __name__ == "__main__":
    main()
