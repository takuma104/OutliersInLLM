"""Phase 2 comparison across runs: training curves, KL-vs-outlier Pareto plots and a Markdown summary table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
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
        undo = next((v for k, v in e.items() if k.startswith("neutralize_") and k.endswith("heldout_kl")), None)
        row |= {"WT2 PPL": e["wikitext2/ppl"], "eval KL": e["heldout/kl"], "eval M2": e["outlier/sink_score_median"],
                "eval p50": e["outlier/peak_p50_median"], "W4A4": e.get("rtn/W4A4"), "A4-INT@qkv": e.get("rtn/A4-INT@qkv"),
                "A4-INT@gate_up": e.get("rtn/A4-INT@gate_up"), "A4-INT": e.get("rtn/A4-INT"),
                "lm-eval avg": e.get("lmeval/avg_acc"), "KL w/o new params": undo}
    return row


EVAL_Y = [("eval p50", "outlier/peak_p50_median", "peak |u| p50 (median over reads)"),
          ("eval M2", "outlier/sink_score_median", "residual sink score M2 (median)"),
          ("A4-INT@gate_up", "rtn/A4-INT@gate_up", "ΔPPL, INT4 per-token acts on gate_up"),
          ("W4A4", "rtn/W4A4", "ΔPPL, W4A4 NVFP4")]
ARM_COLOR = {"orig": C[2], "GatedNorm": C[0], "bias": C[1]}
LAM_MARKER = ["o", "s", "^", "D"]


def fig_pareto(table: pd.DataFrame, base_eval: dict[str, float] | None, out: Path) -> None:
    """Held-out KL vs outlier / quantization metrics from the checkpoint evaluations; original model at KL = 0."""
    t = table.dropna(subset=["eval KL"])
    if t.empty:
        return
    lams = sorted(t["λ"].unique())
    fig, axes = plt.subplots(1, len(EVAL_Y), figsize=(3.4 * len(EVAL_Y), 3.2))
    for ax, (col, base_key, label) in zip(axes, EVAL_Y):
        if base_eval is not None and base_key in base_eval:
            ax.scatter([0], [base_eval[base_key]], color=GRAY, marker="*", s=90, zorder=3, label="original")
        for i, arm in enumerate(sorted(t.arm.unique())):
            sub = t[t.arm == arm].sort_values("λ")
            color = ARM_COLOR.get(arm, C[(i + 3) % len(C)])
            ax.plot(sub["eval KL"], sub[col], color=color, linewidth=1.2, label=arm)
            for _, row in sub.iterrows():
                ax.scatter(row["eval KL"], row[col], color=color, s=36, zorder=3,
                           marker=LAM_MARKER[lams.index(row["λ"]) % len(LAM_MARKER)])
        ax.set_xlim(left=-0.0005)
        ax.set_xlabel("held-out KL(orig ‖ θ)")
        ax.set_title(label)
        if col.startswith(("A4", "W4")):
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    handles, labels = axes[0].get_legend_handles_labels()
    for j, lam in enumerate(lams):
        handles.append(plt.Line2D([], [], color=GRAY, marker=LAM_MARKER[j % len(LAM_MARKER)], linestyle="none"))
        labels.append(f"λ = {lam:g}")
    axes[0].legend(handles, labels, fontsize=7)
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
    ap.add_argument("--base-eval", type=Path, default=None, help="summary.json of the original model")
    args = ap.parse_args()
    setup_style()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.root, args.pattern)
    base_metrics = runs.metrics.iloc[0].iloc[0]  # step-0 probe of any run = original model
    base = {k: float(base_metrics[f"probe/{k}"]) for k, _ in PARETO_Y}
    fig_curves(runs, args.out, args.color_by)
    table = pd.DataFrame([final_row(r) for _, r in runs.iterrows()])
    base_eval = json.loads(args.base_eval.read_text()) if args.base_eval else None
    fig_pareto(table, base_eval, args.out)
    print("original (step-0 probe):", {k: round(v, 3) for k, v in base.items()})
    print(md(table))
    (args.out / "table.md").write_text(md(table) + "\n")


if __name__ == "__main__":
    main()
