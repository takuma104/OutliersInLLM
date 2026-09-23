"""Phase 1 §4.1 E: cheap causal probe of the residual sink as a rescaling lever.

For each hidden-size RMSNorm, the sink dim d of its *input* (from Phase 1 residual stats) is intervened on
only in what the norm sees (the residual stream itself is untouched), for non-first tokens:
  mean : x_d <- E[x_d]        (input-independent constant; keeps the norm denominator on average)
  zero : x_d <- 0             (removes the lever: the other dims get rescaled)
  clamp: |x_d| <= τ·rms(x_{-d}) (caps the peak |u_d| near τ, what the Phase 2 regularizer would enforce)
If mean is harmless but zero breaks the model, the sink acts as an input-independent rescale factor.

Zero ablation is further decomposed on the norm *output* y = λ·x/rms(x) (matters when λ_d is not ≈ 0):
  direct_zero: y_d <- 0, other dims untouched             (only the sink's own feature is removed)
  denom_zero : y_{j≠d} <- y_j · rms(x)/rms(x with x_d=0)   (only the rescaling lever is removed)

Runs (a) all norms at once and (b) one residual read at a time, reporting ΔPPL on WikiText-2.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import torch
import wandb
from torch import nn

from outliers.data import chunk_tokens, token_byte_lengths, wikitext2_test_text
from outliers.evaluate import evaluate_lm
from outliers.models import iter_residual_norms, load_model

WANDB_PROJECT = "OutliersInLLM"


def ablate(x: torch.Tensor, dim: int, mode: str, mean: float, tau: float, include_first: bool) -> torch.Tensor:
    y = x.clone()
    sl = slice(None) if include_first else slice(1, None)
    col = y[:, sl, dim]
    if mode == "mean":
        col.fill_(mean)
    elif mode == "zero":
        col.zero_()
    elif mode == "clamp":
        rest = torch.cat([y[:, sl, :dim], y[:, sl, dim + 1 :]], dim=-1).float()
        lim = tau * rest.square().mean(-1).sqrt()
        y[:, sl, dim] = torch.maximum(torch.minimum(col.float(), lim), -lim).to(y.dtype)
    else:
        raise ValueError(mode)
    return y


def ablate_output(x: torch.Tensor, y: torch.Tensor, dim: int, mode: str, include_first: bool) -> torch.Tensor:
    y = y.clone()
    sl = slice(None) if include_first else slice(1, None)
    if mode == "direct_zero":
        y[:, sl, dim] = 0
    elif mode == "denom_zero":
        xf = x[:, sl].float()
        ss = xf.square().sum(-1, keepdim=True)
        d = x.shape[-1]
        scale = torch.sqrt((ss / d + 1e-6) / ((ss - xf[..., dim : dim + 1].square()) / d + 1e-6))
        keep = y[:, sl, dim].clone()
        y[:, sl] = (y[:, sl].float() * scale).to(y.dtype)
        y[:, sl, dim] = keep
    else:
        raise ValueError(mode)
    return y


OUTPUT_MODES = ("direct_zero", "denom_zero")


@contextlib.contextmanager
def sink_intervention(
    targets: list[tuple[nn.Module, int, float]], mode: str, tau: float, include_first: bool
) -> Iterator[None]:
    handles = []
    for mod, dim, mean in targets:
        if mode in OUTPUT_MODES:

            def out_hook(_m: nn.Module, args: tuple[torch.Tensor, ...], y: torch.Tensor, dim: int = dim) -> torch.Tensor:
                return ablate_output(args[0], y, dim, mode, include_first)

            handles.append(mod.register_forward_hook(out_hook))
        else:

            def hook(_m: nn.Module, args: tuple[torch.Tensor, ...], dim: int = dim, mean: float = mean) -> tuple:
                return (ablate(args[0], dim, mode, mean, tau, include_first),) + args[1:]

            handles.append(mod.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", type=Path, default=Path("results/phase1"))
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--windows-all", type=int, default=None, help="WikiText windows for all-norm runs")
    ap.add_argument("--windows-single", type=int, default=24, help="WikiText windows for per-norm runs")
    ap.add_argument("--taus", type=float, nargs="+", default=[4.0, 8.0])
    ap.add_argument("--include-first", action="store_true")
    ap.add_argument("--no-single", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    model, tok = load_model(args.model)
    windows = chunk_tokens(wikitext2_test_text(), tok, args.seq_len)
    bl = token_byte_lengths(tok)
    residual = pd.read_parquet(args.root / args.model / "residual.parquet")
    dims = pd.read_parquet(args.root / args.model / "residual_dims.parquet")
    residual = residual[residual.kind == "residual"].set_index("name")
    dims = dims[dims.site.str.endswith("norm_in")]

    targets: dict[str, tuple[nn.Module, int, float]] = {}
    for ni in iter_residual_norms(model):
        d = int(residual.loc[ni.name, "sink_dim"])
        mean = float(dims[(dims.name == ni.name) & (dims.dim == d)]["mean"].iloc[0])
        targets[ni.name] = (ni.module, d, mean)

    modes = [("mean", 0.0), ("zero", 0.0), ("direct_zero", 0.0), ("denom_zero", 0.0)] + [
        ("clamp", t) for t in args.taus
    ]
    rows: list[dict[str, object]] = []

    def evaluate(scope: str, tgts: list[tuple[nn.Module, int, float]], mode: str, tau: float, n: int | None) -> None:
        w = windows[:n]
        if mode == "none":
            r = evaluate_lm(model, w, bl)
        else:
            with sink_intervention(tgts, mode, tau, args.include_first):
                r = evaluate_lm(model, w, bl)
        rows.append({"scope": scope, "mode": mode, "tau": tau, "n_windows": len(w)} | r)
        print(f"[{time.perf_counter() - t0:6.0f}s] {scope:42s} {mode:11s} τ={tau:<4} ppl={r['ppl']:.4f}", flush=True)

    evaluate("none", [], "none", 0.0, args.windows_all)
    for mode, tau in modes:
        evaluate("all", list(targets.values()), mode, tau, args.windows_all)
    if not args.no_single:
        evaluate("none", [], "none", 0.0, args.windows_single)
        for name, tgt in targets.items():
            for mode, tau in modes:
                evaluate(name, [tgt], mode, tau, args.windows_single)

    df = pd.DataFrame(rows)
    base = df[df["mode"] == "none"].groupby("n_windows")["ppl"].first()
    df["dppl_rel"] = df.ppl / df.n_windows.map(base) - 1
    df["sink_dim"] = df.scope.map(lambda s: targets[s][1] if s in targets else -1)
    suffix = "_withfirst" if args.include_first else ""
    df.to_parquet(args.root / args.model / f"ablate_sink{suffix}.parquet")
    print(df[df.scope.isin(["none", "all"])][["scope", "mode", "tau", "ppl", "dppl_rel"]].to_string())

    if not args.no_wandb:
        run = wandb.init(project=WANDB_PROJECT, group="phase1", job_type="ablate_sink",
                         name=f"phase1-ablate-{args.model}{suffix}", config=vars(args) | {"model_key": args.model})
        run.log({"ablate_sink": wandb.Table(dataframe=df)})
        run.finish()


if __name__ == "__main__":
    main()
