"""Phase 1 follow-up: does per-token INT4 at the step-up MLP input break the massive activation?

Quantizes the gate/up input of one MLP to INT4 (per token) for all tokens, only the first token, or only the other
tokens, and reports WikiText-2 PPL and the first token's max |h| entering the next layer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from outliers.data import chunk_tokens, token_byte_lengths, wikitext2_test_text
from outliers.evaluate import evaluate_lm
from outliers.models import load_model
from outliers.quant import quant_int


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--windows", type=int, default=24)
    ap.add_argument("--root", type=Path, default=Path("results/phase1"))
    args = ap.parse_args()

    model, tok = load_model(args.model)
    windows = chunk_tokens(wikitext2_test_text(), tok, 2048)[: args.windows]
    bl = token_byte_lengths(tok)
    mlp = model.model.layers[args.layer].mlp
    nxt = model.model.layers[args.layer + 1].input_layernorm
    base = evaluate_lm(model, windows, bl)["ppl"]
    rows = []
    for which in ("all", "first", "rest"):
        sl = {"all": slice(None), "first": slice(0, 1), "rest": slice(1, None)}[which]

        def hook(_m: torch.nn.Module, args_: tuple[torch.Tensor, ...], sl: slice = sl) -> tuple[torch.Tensor, ...]:
            x = args_[0].clone()
            x[:, sl] = quant_int(x[:, sl], 4)
            return (x,)

        first_max: list[float] = []
        handles = [mlp.gate_proj.register_forward_pre_hook(hook), mlp.up_proj.register_forward_pre_hook(hook),
                   nxt.register_forward_pre_hook(lambda _m, a: first_max.append(float(a[0][:, 0].abs().max())))]
        ppl = evaluate_lm(model, windows, bl)["ppl"]
        for h in handles:
            h.remove()
        rows.append({"tokens": which, "ppl": ppl, "dppl_rel": ppl / base - 1, "first_token_max_next": max(first_max)})
    df = pd.DataFrame(rows)
    df.to_parquet(args.root / args.model / f"stepup_quant_L{args.layer}.parquet")
    print(f"base ppl {base:.3f}\n{df.to_string()}")


if __name__ == "__main__":
    main()
