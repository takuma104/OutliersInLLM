"""Phase 3 summary: markdown tables and figures for the report from results/phase3/*.

Usage: python scripts/phase3_summary.py  (writes docs/reports/figs/phase3/*.png and prints the tables)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("results/phase3")
FIGS = Path("docs/reports/figs/phase3")
MODELS = ("M0", "M1", "M3")
MODEL_LABEL = {"M0": "M0 original", "M1": "M1 original arch. (B1)", "M3": "M3 GA+GatedNorm (B3h)"}
MODEL_COLOR = {"M0": "#6b7280", "M1": "#d97706", "M3": "#2563eb"}
METHODS = ("rtn-ternary", "gptq-ternary", "catq", "gptq-int4")
# FP lm-eval references come from the Phase 2 evaluations of the same checkpoints
PHASE2_EVAL = {"M0": Path("results/phase2/base-qwen3-0.6b/eval"), "M1": Path("results/phase2/c2/B1-lam3e-3/eval"),
               "M3": Path("results/phase2/c2/B3h-lam3e-3/eval")}
# CAT-Q paper protocol: PIQA / ARC-e / ARC-c / HellaSwag acc_norm, WinoGrande acc
PAPER5 = ("piqa/acc_norm", "arc_easy/acc_norm", "arc_challenge/acc_norm", "hellaswag/acc_norm", "winogrande/acc")


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def collect() -> pd.DataFrame:
    rows = []
    for m in MODELS:
        fp = _json(ROOT / f"fp-{m}/eval/summary.json")
        for method in ("fp", *METHODS):
            d = ROOT / (f"fp-{m}" if method == "fp" else f"{m}-{method}")
            s = _json(d / "eval/summary.json")
            if s is None:
                continue
            dg = _json(d / "diagnose/summary.json") or {}
            lm = _json(PHASE2_EVAL[m] / "summary.json") if method == "fp" else s
            paper5 = [lm.get(f"lmeval/{k}") for k in PAPER5] if lm else []
            rows.append({
                "model": m, "method": method,
                "wt2": s["wikitext2/ppl"], "c4val8": s["c4val8/ppl"], "c4val64": s["c4val64/ppl"],
                "c4val64_x_fp": s["c4val64/ppl"] / fp["c4val64/ppl"] if fp else None,
                "kl": s["heldout/kl"],
                "A8": s.get("rtn/A8"), "A4": s.get("rtn/A4"), "A4-INT": s.get("rtn/A4-INT"),
                "max_first": s["outlier/max_first"], "sink": s.get("attn/sink_ratio"),
                "top1": dg.get("top1_agree"), "top1_first": dg.get("top1_agree_first"),
                "err_last": dg.get("last/pre_rest"), "err_last_first": dg.get("last/pre_first"),
                "ma_ratio_min": dg.get("min/ma_amp_ratio"),
                "avg5": sum(paper5) / 5 if paper5 and None not in paper5 else None,
                "lambada": lm.get("lmeval/lambada_openai/acc") if lm else None,
                **{k: lm.get(f"lmeval/{k}") if lm else None for k in PAPER5},
            })
    return pd.DataFrame(rows)


def fmt_ppl(x: float) -> str:
    return f"{x:,.0f}" if x >= 1000 else (f"{x:.1f}" if x >= 100 else f"{x:.2f}")


def table_main(df: pd.DataFrame) -> str:
    lines = ["| 方式 | モデル | WikiText-2 | C4-val8 | C4-val64 | ×自分の FP | held-out KL | top-1 一致 | MA（先頭） |",
             "|---|---|---|---|---|---|---|---|---|"]
    for method in ("fp", *METHODS):
        for _, r in df[df.method == method].iterrows():
            x = "–" if method == "fp" else f"{r.c4val64_x_fp:.2f}"
            t1 = "–" if pd.isna(r.top1) else f"{r.top1:.3f}"
            lines.append(f"| {method} | {r.model} | {fmt_ppl(r.wt2)} | {fmt_ppl(r.c4val8)} | {fmt_ppl(r.c4val64)} | {x} "
                         f"| {r.kl:.3f} | {t1} | {r.max_first:.0f} |")
    return "\n".join(lines)


def table_act(df: pd.DataFrame) -> str:
    lines = ["| 方式 | モデル | A8 | A4（NVFP4） | A4-INT（per-token） |", "|---|---|---|---|---|"]
    for method in ("fp", *METHODS):
        for _, r in df[df.method == method].iterrows():
            lines.append(f"| {method} | {r.model} | {r.A8:+.1%} | {r.A4:+.1%} | {r['A4-INT']:+.0%} |")
    return "\n".join(lines)


def table_lmeval(df: pd.DataFrame) -> str:
    lines = ["| 方式 | モデル | PIQA | ARC-e | ARC-c | HellaSwag | WinoGrande | **5 タスク平均** | LAMBADA |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df[df.avg5.notna()].iterrows():
        cells = " | ".join(f"{100 * r[k]:.1f}" for k in PAPER5)
        lines.append(f"| {r.method} | {r.model} | {cells} | **{100 * r.avg5:.1f}** | {100 * r.lambada:.1f} |")
    return "\n".join(lines)


def fig_ratio(df: pd.DataFrame) -> None:
    methods = [m for m in METHODS if m in set(df.method)]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.26
    for i, m in enumerate(MODELS):
        sub = df[df.model == m].set_index("method")
        xs = [j + (i - 1) * w for j in range(len(methods))]
        ys = [sub.c4val64_x_fp.get(meth, float("nan")) for meth in methods]
        ax.bar(xs, ys, width=w - 0.02, color=MODEL_COLOR[m], label=MODEL_LABEL[m])
    ax.set_yscale("log")
    ax.set_xticks(range(len(methods)), methods)
    ax.set_ylabel("C4-val64 PPL / own FP PPL")
    ax.axhline(1, color="#9ca3af", lw=0.8)
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "ppl_ratio.png", dpi=150)
    plt.close(fig)


def fig_layers(method: str) -> None:
    paths = {m: ROOT / f"{m}-{method}/diagnose/layers.parquet" for m in MODELS}
    if not any(p.exists() for p in paths.values()):
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for m, p in paths.items():
        if not p.exists():
            continue
        g = pd.read_parquet(p).groupby("layer").mean(numeric_only=True)
        kw = {"color": MODEL_COLOR[m], "lw": 2, "label": MODEL_LABEL[m]}
        axes[0].plot(g.index, g.pre_rest, **kw)
        axes[1].plot(g.index, g.pre_first, **kw)
        axes[2].plot(g.index, g.ma_amp_ratio, **kw)
    axes[0].set_title("rel. error of h (positions ≥ 1)", fontsize=9)
    axes[1].set_title("rel. error of h (position 0)", fontsize=9)
    axes[2].set_title("largest position-0 channel: h_q / h_fp", fontsize=9)
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("layer")
        ax.grid(color="#e5e7eb", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{method}: error vs the model's own FP (C4-val8)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / f"layers_{method}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(FIGS / "summary.csv", index=False)
    print(table_main(df), "\n")
    print(table_act(df), "\n")
    print(table_lmeval(df), "\n")
    fig_ratio(df)
    for method in METHODS:
        fig_layers(method)


if __name__ == "__main__":
    main()
