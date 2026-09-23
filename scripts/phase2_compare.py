"""Phase 2 comparison across runs: training curves, KL-vs-outlier Pareto plots and a Markdown summary table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# reference categorical palette (validated, light mode), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#8a8984"
LINESTYLES = ["-", (0, (4, 2)), (0, (1.5, 1.5)), (0, (6, 2, 1.5, 2))]
CURVES = [("heldout/kl", "held-out KL(orig ‖ θ)", True), ("train/R", "R (train)", True),
          ("probe/peak_p50_median", "peak |u| p50 (median over reads)", False),
          ("probe/peak_p99_max", "peak |u| p99 (max over reads)", False)]
PARETO_Y = [("peak_p50_median", "peak |u| p50, median over reads"), ("peak_p99_max", "peak |u| p99, max over reads"),
            ("sink_score_median", "residual sink score (M2), median")]


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb", "font.size": 8.5, "axes.titlesize": 9.5, "axes.edgecolor": "#b9b8b3",
        "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
        "grid.linewidth": 0.7, "lines.linewidth": 1.5, "legend.frameon": False, "legend.fontsize": 7.5,
    })


def arm_of(cfg: dict) -> str:
    r = cfg["retrofit"]
    parts = []
    if r.get("attn_gate"):
        parts.append(f"GA-{r['attn_gate'][:4]}")
    if r.get("gated_norm"):
        parts.append("GatedNorm")
    if r.get("linear_bias"):
        parts.append("bias")
    arch = "+".join(parts) or "orig"
    return arch + (" (new only)" if cfg.get("new_only") else "")


def load_runs(root: Path, pattern: str) -> pd.DataFrame:
    rows = []
    for d in sorted(root.glob(pattern)):
        if not (d / "metrics.jsonl").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        m = pd.read_json(d / "metrics.jsonl", lines=True)
        rows.append({"run": cfg["run"], "dir": d, "arm": arm_of(cfg), "lam": cfg["lam"], "lr": cfg["lr"],
                     "metrics": m, "eval": json.loads((d / "eval" / "summary.json").read_text())
                     if (d / "eval" / "summary.json").exists() else None})
    return pd.DataFrame(rows)


def fig_curves(runs: pd.DataFrame, out: Path, color_by: str) -> None:
    keys = sorted(runs[color_by].unique(), key=str)
    styles = sorted(runs["lam" if color_by == "arm" else "arm"].unique(), key=str)
    fig, axes = plt.subplots(1, len(CURVES), figsize=(3.3 * len(CURVES), 3.1))
    for r in runs.itertuples():
        c = C[keys.index(getattr(r, color_by)) % len(C)]
        ls = LINESTYLES[styles.index(r.lam if color_by == "arm" else r.arm) % len(LINESTYLES)]
        for ax, (k, label, logy) in zip(axes, CURVES):
            s = r.metrics[["step", k]].dropna()
            if k == "heldout/kl":
                s = s[s.step > 0]
            ax.plot(s.step, s[k], color=c, linestyle=ls, label=f"{r.arm}, λ={r.lam:g}")
            ax.set_title(label)
            if logy:
                ax.set_yscale("log")
    for ax in axes:
        ax.set_xlabel("step")
    axes[0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "curves.png")
    plt.close(fig)


def final_row(r: pd.Series) -> dict[str, float]:
    last = r.metrics.dropna(subset=["probe/peak_p50_median"]).iloc[-1]
    row = {"run": r.run, "arm": r.arm, "λ": r.lam, "steps": int(last.step), "held-out KL": last["heldout/kl"],
           "R (last step)": r.metrics["train/R"].dropna().iloc[-1]}
    for k, _ in PARETO_Y:
        row[k] = last[f"probe/{k}"]
    for k in ("down_proj_absmax_max", "qkv_chrms_median", "gate_up_chrms_median", "out_proj_chrms_median"):
        if f"probe/{k}" in last:
            row[k] = last[f"probe/{k}"]
    if r.eval is not None:
        e = r.eval
        row |= {"WT2 PPL": e["wikitext2/ppl"], "RTN W4A4": e.get("rtn/W4A4"), "RTN A4-INT": e.get("rtn/A4-INT"),
                "lm-eval avg": e.get("lmeval/avg_acc")}
    return row


def fig_pareto(table: pd.DataFrame, base: dict[str, float], out: Path) -> None:
    arms = sorted(table.arm.unique())
    fig, axes = plt.subplots(1, len(PARETO_Y), figsize=(3.6 * len(PARETO_Y), 3.2))
    for ax, (k, label) in zip(axes, PARETO_Y):
        ax.scatter([1e-5], [base[k]], color=GRAY, marker="*", s=80, zorder=3, label="original")
        for i, arm in enumerate(arms):
            sub = table[table.arm == arm].sort_values("λ")
            ax.plot(sub["held-out KL"], sub[k], color=C[i % len(C)], marker="o", markersize=5, label=arm)
            for _, row in sub.iterrows():
                ax.annotate(f"λ={row['λ']:g}", (row["held-out KL"], row[k]), xytext=(4, 3),
                            textcoords="offset points", fontsize=6.5, color="#52514e")
        ax.set_xscale("log")
        ax.set_xlabel("held-out KL(orig ‖ θ)  (original at 1e-5)")
        ax.set_title(label)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "pareto.png")
    plt.close(fig)


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = [format(v, ".4g") if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True, help="glob under results/phase2, e.g. 'c1-pilot/*'")
    ap.add_argument("--root", type=Path, default=Path("results/phase2"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--color-by", choices=["arm", "lam"], default="arm")
    args = ap.parse_args()
    setup_style()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.root, args.pattern)
    base_metrics = runs.metrics.iloc[0].iloc[0]  # step-0 probe of any run = original model
    base = {k: float(base_metrics[f"probe/{k}"]) for k, _ in PARETO_Y}
    fig_curves(runs, args.out, args.color_by)
    table = pd.DataFrame([final_row(r) for _, r in runs.iterrows()])
    fig_pareto(table, base, args.out)
    print("original (step-0 probe):", {k: round(v, 3) for k, v in base.items()})
    print(md(table))
    (args.out / "table.md").write_text(md(table) + "\n")


if __name__ == "__main__":
    main()
