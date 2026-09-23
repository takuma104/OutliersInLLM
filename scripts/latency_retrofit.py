"""Phase 2 §5.5: inference overhead of the retrofits (prefill and decode latency relative to the original model).

New parameters are set to small random values (not identity) so no kernel can shortcut them; run on an idle GPU.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from outliers.models import load_model
from outliers.retrofit import RetrofitConfig, apply_retrofit


@torch.no_grad()
def bench(model: torch.nn.Module, prompt_len: int, decode_tokens: int, reps: int) -> dict[str, float]:
    ids = torch.randint(0, 100_000, (1, prompt_len), device="cuda")

    def prefill() -> None:
        model(input_ids=ids, use_cache=False)

    def decode() -> None:
        model.generate(ids[:, :128], max_new_tokens=decode_tokens, min_new_tokens=decode_tokens, do_sample=False)

    out = {}
    for name, fn in (("prefill_ms", prefill), ("decode_ms_per_token", decode)):
        fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps * 1000
        out[name] = dt / decode_tokens if name.startswith("decode") else dt
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("results/phase2/latency"))
    args = ap.parse_args()

    configs = {"orig": RetrofitConfig(), "gatednorm": RetrofitConfig(gated_norm=True),
               "bias": RetrofitConfig(linear_bias=True)}
    if args.model.startswith("qwen3-"):
        configs |= {"ga_headwise": RetrofitConfig(attn_gate="headwise"),
                    "ga_elementwise": RetrofitConfig(attn_gate="elementwise"),
                    "ga_headwise+gatednorm": RetrofitConfig(attn_gate="headwise", gated_norm=True)}
    rows = {}
    for name, cfg in configs.items():
        model, _ = load_model(args.model)
        new = apply_retrofit(model, cfg)
        params = dict(model.named_parameters())
        for n in new:
            params[n].data.normal_(0, 0.01)
        rows[name] = bench(model, args.prompt_len, args.decode_tokens, args.reps)
        del model
        torch.cuda.empty_cache()
    base = rows["orig"]
    for r in rows.values():
        r["prefill_ratio"] = r["prefill_ms"] / base["prefill_ms"]
        r["decode_ratio"] = r["decode_ms_per_token"] / base["decode_ms_per_token"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.model}.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
