"""Phase 1 §4.1 F follow-up: activation-only fake quantization of one (layer, kind) at a time.

Localizes which layers' Linear inputs make a per-kind activation quantization collapse.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from outliers.data import chunk_tokens, token_byte_lengths, wikitext2_test_text
from outliers.evaluate import evaluate_lm
from outliers.models import NON_QUANT_KINDS, iter_linears, load_model
from outliers.quant import SCHEMES, fake_quantize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scheme", default="A4-INT")
    ap.add_argument("--kinds", nargs="+", default=None)
    ap.add_argument("--windows", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--root", type=Path, default=Path("results/phase1"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    model, tok = load_model(args.model)
    windows = chunk_tokens(wikitext2_test_text(), tok, args.seq_len)[: args.windows]
    bl = token_byte_lengths(tok)
    linears = [li for li in iter_linears(model) if li.kind not in NON_QUANT_KINDS]
    kinds = args.kinds or sorted({li.kind for li in linears})
    base = evaluate_lm(model, windows, bl)
    rows = []
    for kind in kinds:
        for layer in sorted({li.layer for li in linears if li.kind == kind}):
            names = {li.name for li in linears if li.kind == kind and li.layer == layer}
            with fake_quantize(model, SCHEMES[args.scheme], kinds=[kind], name_filter=names.__contains__):
                r = evaluate_lm(model, windows, bl)
            rows.append({"scheme": args.scheme, "kind": kind, "layer": layer, "modules": ",".join(sorted(names))} | r)
            print(f"[{time.perf_counter() - t0:5.0f}s] {kind:10s} L{layer:<2d} ppl={r['ppl']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df["dppl_rel"] = df.ppl / base["ppl"] - 1
    df.to_parquet(args.root / args.model / f"rtn_layers_{args.scheme}.parquet")
    top = df.nlargest(8, "dppl_rel")[["kind", "layer", "ppl", "dppl_rel"]]
    print(f"base ppl {base['ppl']:.3f}\n{top.to_string()}")


if __name__ == "__main__":
    main()
