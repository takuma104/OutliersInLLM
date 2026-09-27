"""Phase 3 H5c: where does weight quantization error build up, and does the RMSNorm amplify it?

The quantized model and its own FP source run on the CAT-Q C4-val8 windows. For every decoder layer output h
(the residual stream), per token:
  - pre-norm relative error  ‖h_q − h_fp‖ / ‖h_fp‖
  - post-norm relative error ‖u_q − u_fp‖ / ‖u_fp‖ with u = h / rms(h) (the RMSNorm without its weight)
  - the amplitude ratio h_q[c] / h_fp[c] of the largest first-token channel c of h_fp (regression dilution of the
    massive activation), and the first-token peak |u| of both
Position 0 (the sink token) and the other positions are reported separately. At the output: top-1 agreement and
KL(p_fp ‖ p_q).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from outliers.data import c4_validation_windows
from outliers.models import decoder_layers, load_model
from outliers.retrofit import load_retrofit


def load(src: str) -> tuple[nn.Module, object]:
    if src.endswith(".pt"):
        model, tok, *_ = load_retrofit(Path(src))
        return model, tok
    return load_model(src)


@torch.no_grad()
def layer_outputs(model: nn.Module, ids: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
    hs: list[torch.Tensor] = []

    def hook(_m: nn.Module, _a: tuple, out: torch.Tensor | tuple) -> None:
        hs.append((out[0] if isinstance(out, tuple) else out)[0].float())

    handles = [layer.register_forward_hook(hook) for layer in decoder_layers(model)]
    try:
        logits = model(input_ids=ids, use_cache=False).logits[0].float()
    finally:
        for h in handles:
            h.remove()
    return hs, logits


def rel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-12)


def unit(h: torch.Tensor) -> torch.Tensor:
    return h / h.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)


@torch.no_grad()
def diagnose(fp: nn.Module, q: nn.Module, windows: torch.Tensor) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, float]] = []
    top1, kls, top1_first = [], [], []
    for s, ids in enumerate(windows):
        ids = ids.unsqueeze(0).cuda()
        h_fp, lg_fp = layer_outputs(fp, ids)
        h_q, lg_q = layer_outputs(q, ids)
        for li, (a, b) in enumerate(zip(h_q, h_fp, strict=True)):
            pre, post = rel(a, b), rel(unit(a), unit(b))
            c = int(b[0].abs().argmax())
            rows.append({"seq": s, "layer": li,
                         "pre_first": float(pre[0]), "post_first": float(post[0]),
                         "pre_rest": float(pre[1:].mean()), "post_rest": float(post[1:].mean()),
                         "ma_dim": c, "ma_amp_fp": float(b[0, c]), "ma_amp_ratio": float(a[0, c] / b[0, c]),
                         "peak_first_fp": float(unit(b[0]).abs().max()), "peak_first_q": float(unit(a[0]).abs().max()),
                         "peak_rest_fp": float(unit(b[1:]).abs().amax(-1).median()),
                         "peak_rest_q": float(unit(a[1:]).abs().amax(-1).median())})
        agree = lg_q.argmax(-1) == lg_fp.argmax(-1)
        top1.append(float(agree[1:].float().mean()))
        top1_first.append(float(agree[0]))
        kls.append(float(F.kl_div(F.log_softmax(lg_q, -1), F.log_softmax(lg_fp, -1), log_target=True,
                                  reduction="batchmean")))
    df = pd.DataFrame(rows)
    per_layer = df.groupby("layer").mean(numeric_only=True)
    last = per_layer.iloc[-1]
    summary = {"top1_agree": sum(top1) / len(top1), "top1_agree_first": sum(top1_first) / len(top1_first),
               "kl_fp_q": sum(kls) / len(kls),
               "last/pre_rest": float(last.pre_rest), "last/post_rest": float(last.post_rest),
               "last/pre_first": float(last.pre_first), "last/post_first": float(last.post_first),
               "max/post_over_pre_rest": float((per_layer.post_rest / per_layer.pre_rest).max()),
               "min/ma_amp_ratio": float(per_layer.ma_amp_ratio.min())}
    return df, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="results/phase3/<name>/quantized.pt")
    ap.add_argument("--fp", default=None, help="FP source (default: quant_log.json 'source')")
    ap.add_argument("--num-docs", type=int, default=8)
    args = ap.parse_args()
    run_dir = args.ckpt.parent
    fp_src = args.fp or json.loads((run_dir / "quant_log.json").read_text())["source"]
    q, tok = load(str(args.ckpt))
    fp, _ = load(fp_src)
    df, summary = diagnose(fp, q, c4_validation_windows(tok, args.num_docs))
    out = run_dir / "diagnose"
    out.mkdir(exist_ok=True)
    df.to_parquet(out / "layers.parquet")
    (out / "summary.json").write_text(json.dumps(summary | {"fp": fp_src}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
