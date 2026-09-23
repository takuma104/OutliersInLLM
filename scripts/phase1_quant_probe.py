"""Phase 1 §4.1 F: RTN fake-quant ΔPPL sweep on WikiText-2.

Not a PTQ-method comparison: checks whether the outlier metrics line up with quantization difficulty.
- full-model schemes: W8A8 (INT8), W8A8-FP8, W4A16 (INT4 g128), W4A8, W4A4 (NVFP4) and activation-only A8/A4/A4-INT
- bottleneck search: activation-only quantization restricted to one module kind at a time
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import wandb

from outliers.data import chunk_tokens, token_byte_lengths, wikitext2_test_text
from outliers.evaluate import evaluate_lm
from outliers.models import NON_QUANT_KINDS, iter_linears, load_model
from outliers.quant import SCHEMES, fake_quantize

FULL_SCHEMES = ["W8A8", "W8A8-FP8", "W4A16", "W4A8", "W4A4", "A8", "A4", "A4-INT"]
PER_KIND_SCHEMES = ["A4", "A4-INT"]
SUBSETS = {"out_proj": "linear_attn.out_proj", "o_proj_attn": "self_attn.o_proj"}
WANDB_PROJECT = "OutliersInLLM"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-root", type=Path, default=Path("results/phase1"))
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    model, tok = load_model(args.model)
    windows = chunk_tokens(wikitext2_test_text(), tok, args.seq_len)[: args.max_windows]
    bl = token_byte_lengths(tok)
    kinds = sorted({li.kind for li in iter_linears(model)} - NON_QUANT_KINDS)

    rows: list[dict[str, object]] = []

    def run(scheme: str, kind: str | None, sub: str | None = None) -> None:
        name_filter = None if sub is None else (lambda n: SUBSETS[sub] in n)
        with fake_quantize(model, SCHEMES[scheme], kinds=None if kind is None else [kind], name_filter=name_filter):
            r = evaluate_lm(model, windows, bl, batch_size=args.batch_size)
        label = sub or kind or "all"
        rows.append({"scheme": scheme, "kind": label} | r)
        print(f"[{time.perf_counter() - t0:6.0f}s] {scheme:9s} {label:11s} ppl={r['ppl']:.4f}", flush=True)

    base = evaluate_lm(model, windows, bl, batch_size=args.batch_size)
    rows.append({"scheme": "bf16", "kind": "none"} | base)
    for s in FULL_SCHEMES:
        run(s, None)
    for s in PER_KIND_SCHEMES:
        for k in kinds:
            run(s, k)
        if any(li.kind == "o_proj" and SUBSETS["out_proj"] in li.name for li in iter_linears(model)):
            # the normalized o_proj kind mixes attention o_proj and the GDN out_proj
            for sub in SUBSETS:
                run(s, "o_proj", sub)

    df = pd.DataFrame(rows)
    df["dppl"] = df.ppl - base["ppl"]
    df["dppl_rel"] = df.ppl / base["ppl"] - 1
    df["dnll"] = df.nll - base["nll"]
    out = args.out_root / args.model
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "rtn.parquet")
    print(df[["scheme", "kind", "ppl", "dppl", "dppl_rel"]].to_string())

    if not args.no_wandb:
        run_ = wandb.init(project=WANDB_PROJECT, group="phase1", job_type="rtn", name=f"phase1-rtn-{args.model}",
                          config=vars(args) | {"model_key": args.model})
        run_.log({"rtn": wandb.Table(dataframe=df)})
        run_.summary.update({f"dppl_rel/{r.scheme}/{r.kind}": r.dppl_rel for r in df.itertuples()})
        run_.finish()


if __name__ == "__main__":
    main()
