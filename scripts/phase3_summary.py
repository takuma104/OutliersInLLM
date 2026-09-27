"""Phase 3 summary: markdown tables and figures for the report from results/phase3/*.

Usage: python scripts/phase3_summary.py  (writes docs/reports/figs/phase3/*.png and prints the tables)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def doc_nll(run: str) -> np.ndarray | None:
    p = ROOT / run / "eval/c4val64_doc_nll.npy"
    return np.load(p) if p.exists() else None


BOOT = np.random.default_rng(0).integers(0, 64, size=(10000, 64))


def boot_ci(diff: np.ndarray) -> tuple[float, float, float]:
    """exp(mean per-document NLL difference) with a paired 95% bootstrap interval over the 64 documents."""
    x = np.exp(diff[BOOT].mean(1))
    return float(np.exp(diff.mean())), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def table_pairs(methods: tuple[str, ...] = ("gptq-ternary", "catq", "gptq-int4")) -> str:
    lines = ["| 方式 | 比較 | C4-val64 PPL の比 [95% 区間] | 自分の FP に対する劣化の比 [95% 区間] |", "|---|---|---|---|"]
    for method in methods:
        for a, b in (("M3", "M0"), ("M1", "M0"), ("M3", "M1")):
            qa, qb, fa, fb = doc_nll(f"{a}-{method}"), doc_nll(f"{b}-{method}"), doc_nll(f"fp-{a}"), doc_nll(f"fp-{b}")
            if qa is None or qb is None or fa is None or fb is None:
                continue
            r = boot_ci(qa - qb)
            rel = boot_ci((qa - fa) - (qb - fb))
            lines.append(f"| {method} | {a} / {b} | {r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}] "
                         f"| {rel[0]:.3f} [{rel[1]:.3f}, {rel[2]:.3f}] |")
    return "\n".join(lines)


def _bars(ax: plt.Axes, df: pd.DataFrame, methods: list[str], excess: bool) -> None:
    w = 0.26
    for i, m in enumerate(MODELS):
        sub = df[df.model == m].set_index("method")
        for j, meth in enumerate(methods):
            if meth not in sub.index:
                continue
            x = j + (i - 1) * w
            y = sub.c4val64_x_fp[meth] - (1 if excess else 0)
            ax.bar(x, y, width=w - 0.03, color=MODEL_COLOR[m], label=MODEL_LABEL[m] if j == 0 else None)
            q, f = doc_nll(f"{m}-{meth}"), doc_nll(f"fp-{m}")
            if excess and q is not None and f is not None:
                _, lo, hi = boot_ci(q - f)
                ax.errorbar(x, y, yerr=[[y - (lo - 1)], [(hi - 1) - y]], color="#111827", lw=1, capsize=2)
                ax.text(x, hi - 1, f"{y:.2f}", ha="center", va="bottom", fontsize=7, color="#374151")
    ax.set_xticks(range(len(methods)), methods)
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def fig_ratio(df: pd.DataFrame) -> None:
    present = set(df.method)
    tern = [m for m in ("rtn-ternary", "gptq-ternary") if m in present]
    mild = [m for m in ("catq", "gptq-int4") if m in present]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1]})
    _bars(a1, df, tern, excess=False)
    a1.set_yscale("log")
    a1.set_ylabel("C4-val64 PPL / own FP PPL")
    a1.set_title("(a) cheap ternary quantizers (log scale)", fontsize=9)
    a1.legend(frameon=False, fontsize=7)
    _bars(a2, df, mild, excess=True)
    a2.set_ylabel("C4-val64 PPL / own FP PPL − 1")
    a2.set_title("(b) CAT-Q ternary and GPTQ INT4 (95% CI over documents)", fontsize=9)
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


RECOVER = ROOT / "recover"
ARMS = (("M3-catq", "gates"), ("M3-catq", "norms"), ("M0-catq", "norms"), ("M0-catq", "lora"))
ARM_LABEL = {"gates": "gates + norms", "norms": "norms only", "lora": "LoRA r=4 + norms"}


def table_recover() -> str:
    lines = ["| 元の三値モデル | 学習するもの | パラメータ数 | LR | held-out KL（前 → 後） | C4-val64 PPL（前 → 後） "
             "| 回復率 | 5 タスク平均 | LAMBADA |", "|---|---|---|---|---|---|---|---|---|"]
    for src, train in ARMS:
        run = RECOVER / f"{src}-{train}"
        s, cfg = _json(run / "eval/summary.json"), _json(run / "config.json")
        q, fp = _json(ROOT / src / "eval/summary.json"), _json(ROOT / f"fp-{src[:2]}/eval/summary.json")
        if s is None or cfg is None:
            continue
        rec = (q["c4val64/ppl"] - s["c4val64/ppl"]) / (q["c4val64/ppl"] - fp["c4val64/ppl"])
        avg5 = sum(s[f"lmeval/{k}"] for k in PAPER5) / 5
        lines.append(f"| {src} | {ARM_LABEL[train]} | {cfg['n_train_params'] / 1e6:.2f}M | {cfg['lr']:g} "
                     f"| {q['heldout/kl']:.3f} → {s['heldout/kl']:.3f} | {q['c4val64/ppl']:.2f} → {s['c4val64/ppl']:.2f} "
                     f"| {rec:.0%} | {100 * avg5:.1f} | {100 * s['lmeval/lambada_openai/acc']:.1f} |")
    return "\n".join(lines)


def fig_recover() -> None:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    styles = {"gates": "-", "norms": "--", "lora": ":"}
    drawn = False
    for src, train in ARMS:
        p = RECOVER / f"{src}-{train}/metrics.jsonl"
        if not p.exists():
            continue
        cfg = _json(RECOVER / f"{src}-{train}/config.json")
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        pts = [(r["step"] * cfg["global_batch"] * cfg["seq_len"] / 1e6, r["heldout/kl"]) for r in rows if "heldout/kl" in r]
        ax.plot(*zip(*pts, strict=True), styles[train], color=MODEL_COLOR[src[:2]], lw=2,
                label=f"{src[:2]}: {ARM_LABEL[train]}")
        drawn = True
    if not drawn:
        plt.close(fig)
        return
    ax.set_xlabel("training tokens (M)")
    ax.set_ylabel("held-out KL to M0 FP")
    ax.grid(color="#e5e7eb", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "recover_kl.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(FIGS / "summary.csv", index=False)
    print(table_main(df), "\n")
    print(table_act(df), "\n")
    print(table_lmeval(df), "\n")
    print(table_pairs(), "\n")
    print(table_recover(), "\n")
    fig_recover()
    fig_ratio(df)
    for method in METHODS:
        fig_layers(method)


if __name__ == "__main__":
    main()
