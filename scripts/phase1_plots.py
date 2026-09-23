"""Phase 1 figures from results/phase1/<model>/*.parquet -> docs/reports/figs/phase1/*.png."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

MODELS = {"qwen3.5-0.8b": "Qwen3.5-0.8B-Base", "qwen3-0.6b": "Qwen3-0.6B-Base"}
# reference categorical palette (validated, light mode), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MODEL_COLOR = {"qwen3.5-0.8b": C[0], "qwen3-0.6b": C[1]}
CAT_COLOR = {"first": C[0], "delim": C[1], "other": C[2]}
KIND_ORDER = ["qkv", "in_proj_z", "o_proj", "out_proj", "gate_up", "down_proj"]
KIND_STYLE = {k: (C[i], m) for i, (k, m) in enumerate(zip(KIND_ORDER, ["o", "s", "^", "P", "D", "v"]))}
KIND_LABEL = {"o_proj": "o_proj (attn)", "out_proj": "out_proj (GDN)"}
BLUE_RAMP = ["#184f95", "#2a78d6", "#86b6ef"]
GRAY = "#8a8984"
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.edgecolor": "#b9b8b3",
            "axes.labelcolor": "#52514e",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
            "text.color": "#0b0b0b",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e4e3df",
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


def load(root: Path, model: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(root / model / f"{name}.parquet")


def display_kind(df: pd.DataFrame) -> pd.DataFrame:
    """Separate the GDN out_proj from attention o_proj (both are normalized to 'o_proj' in the stats)."""
    df = df.copy()
    df.loc[df["name"].str.contains("linear_attn.out_proj"), "kind"] = "out_proj"
    return df


def token_max_abs(root: Path, model: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-token max |h| (= peak|u| · rms) and category for every residual read."""
    tv = np.load(root / model / "tokens.npz")
    names = sorted({k.rsplit("/", 1)[0] for k in tv.files})
    return {n: (tv[f"{n}/peak"] * tv[f"{n}/rms"], tv[f"{n}/cat"]) for n in names}


def massive_counts(root: Path, model: str) -> pd.DataFrame:
    """Tokens with a massive activation (Sun et al. 2024: |h| > 100 and > 1000× the median |h|)."""
    r = load(root, model, "residual")
    med = r[r.kind == "residual"].set_index("name")["median_abs"]
    rows = []
    for name, (mx, cat) in token_max_abs(root, model).items():
        thr = max(100.0, 1000.0 * float(med[name]))
        big = mx > thr
        rows.append({"name": name, "threshold": thr, "n_first": int((big & (cat == 0)).sum()),
                     "n_delim": int((big & (cat == 1)).sum()), "n_other": int((big & (cat == 2)).sum()),
                     "n_first_total": int((cat == 0).sum())})
    return pd.DataFrame(rows)


def layer_x(df: pd.DataFrame) -> np.ndarray:
    """Residual read position in layer units: attn-norm of layer l -> l, mlp-norm -> l+0.5."""
    return df["depth"].to_numpy() / 2.0


def fig_depth_profile(root: Path, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex="col")
    for j, (m, title) in enumerate(MODELS.items()):
        r = load(root, m, "residual")
        r = r[r.kind == "residual"].sort_values("depth")
        x = layer_x(r)
        ax = axes[0, j]
        for k, col in zip((1, 2, 3), BLUE_RAMP):
            ax.plot(x, r[f"top{k}"], color=col, label=f"top-{k}")
        ax.plot(x, r["median_abs"], color=GRAY, label="median")
        ax.set_yscale("log")
        ax.set_title(f"{title}: largest |h| in the residual stream")
        ax.legend(ncol=4, loc="upper left")
        ax = axes[1, j]
        for cat, col in CAT_COLOR.items():
            ax.plot(x, r[f"max_{cat}"], color=col, marker="o", markersize=2.5, label=f"{cat} token (max)")
        tm = token_max_abs(root, m)
        p999 = [float(np.quantile(tm[n][0][tm[n][1] == 2], 0.999)) for n in r["name"]]
        ax.plot(x, p999, color=CAT_COLOR["other"], linestyle=(0, (1.5, 1.5)), label="other token (p99.9)")
        ax.set_yscale("log")
        ax.set_title("max |h| by token type")
        ax.set_xlabel("layer (x.0 = before token mixer, x.5 = before MLP)")
        ax.legend(loc="upper left")
    axes[0, 0].set_ylabel("|h|")
    axes[1, 0].set_ylabel("|h|")
    fig.tight_layout()
    fig.savefig(out / "depth_profile.png")
    plt.close(fig)


def fig_block_outputs(root: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for j, (m, title) in enumerate(MODELS.items()):
        r = load(root, m, "residual")
        b = r[r.kind == "block"].sort_values("depth")
        ax = axes[j]
        for cat, col in CAT_COLOR.items():
            ax.plot(b["depth"] / 2.0, b[f"max_{cat}"], color=col, marker="o", markersize=2.5, label=f"{cat} token")
        ax.set_yscale("log")
        ax.set_title(f"{title}: block output max |·|")
        ax.set_xlabel("layer (x.0 = token mixer out, x.5 = MLP out)")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("max |block output|")
    fig.tight_layout()
    fig.savefig(out / "block_outputs.png")
    plt.close(fig)


def fig_residual_sink(root: Path, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.0), sharex=True)
    for m, title in MODELS.items():
        r = load(root, m, "residual")
        r = r[r.kind == "residual"].sort_values("depth")
        norms = load(root, m, "norms")
        x = layer_x(r)
        col = MODEL_COLOR[m]
        axes[0, 0].plot(x, r["sink_score"], color=col, label=title)
        axes[0, 1].plot(x, r["sink_argmax_frac"], color=col, label=title)
        axes[1, 0].plot(x, r["peak_p50"], color=col, label=f"{title} p50")
        axes[1, 0].plot(x, r["peak_p99"], color=col, linestyle=(0, (1.5, 1.5)), label=f"{title} p99")
        for site, ls, mk in (("attn", "-", "o"), ("mlp", (0, (1.5, 1.5)), "s")):
            rs = r[r["sub"] == site]
            lam_sink = []
            for row in rs.itertuples():
                sub = norms[norms.norm == row.name].set_index("dim")["lambda_eff"]
                lam_sink.append(abs(float(sub.loc[row.sink_dim])))
            label = f"{title}: {'input_layernorm' if site == 'attn' else 'post_attention_layernorm'}"
            axes[1, 1].plot(rs["layer"], lam_sink, color=col, linestyle=ls, marker=mk, markersize=2.5, label=label)
    axes[0, 0].set_title("M2 residual sink score (max/median of mean |h_j|)")
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_title("M2 fraction of tokens whose argmax dim = sink dim")
    axes[0, 1].set_ylim(0, 1.02)
    axes[1, 0].set_title("M3 peak |u_j| of u = x/rms(x) (non-first tokens)")
    axes[1, 0].axhline(np.sqrt(1024), color=GRAY, linewidth=0.8)
    axes[1, 0].text(0.2, np.sqrt(1024) * 0.93, "√d = 32 (upper bound)", color="#52514e", fontsize=7.5, va="top")
    axes[1, 1].set_title("M4 |λ_eff| at the sink dim of each norm input")
    axes[1, 1].set_yscale("log")
    axes[1, 0].set_xlabel("layer (residual read before token mixer / MLP)")
    axes[1, 1].set_xlabel("layer")
    for ax in axes.flat:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "residual_sink.png")
    plt.close(fig)


def fig_sink_dims(root: Path, out: Path) -> None:
    """Per-dim mean |h| and λ_eff at the residual read with the strongest sink."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.2))
    for j, (m, title) in enumerate(MODELS.items()):
        r = load(root, m, "residual")
        r = r[(r.kind == "residual") & (r.sub != "final")]
        row = r.loc[r.sink_score.idxmax()]
        dims = load(root, m, "residual_dims")
        d = dims[(dims.name == row["name"]) & (dims.site == row["site"])].sort_values("dim")
        norms = load(root, m, "norms")
        lam = norms[norms.norm == row["name"]].sort_values("dim")
        sd = int(row.sink_dim)
        ax = axes[0, j]
        ax.vlines(d.dim, 0, d.mean_abs, color=C[0], linewidth=0.8)
        ax.set_yscale("log")
        ax.set_title(f"{title}: mean |h_j| at {row['name'].replace('model.layers.', 'L')}")
        ax.annotate(f"dim {sd}", (sd, float(d.mean_abs.iloc[sd])), xytext=(8, -4), textcoords="offset points",
                    fontsize=7.5)
        ax = axes[1, j]
        ax.vlines(lam.dim, 0, lam.lambda_eff.abs(), color=C[2], linewidth=0.8)
        ax.set_yscale("log")
        ax.set_title(f"|λ_eff| of that norm (dim {sd}: {float(lam.lambda_eff.iloc[sd]):.4f})")
        ax.set_xlabel("hidden dim")
    fig.tight_layout()
    fig.savefig(out / "sink_dims.png")
    plt.close(fig)


def fig_linear_inputs(root: Path, out: Path) -> None:
    metrics = [
        ("absmax", "M5 absmax of input", True),
        ("ch_max_over_median", "M5 channel absmax: max / median", True),
        ("kurt_mean", "M5 per-token excess kurtosis (mean)", True),
        ("sqnr_nvfp4", "M6 activation SQNR, NVFP4 [dB]", False),
        ("sqnr_int4_tok", "M6 activation SQNR, INT4 per-token [dB]", False),
        ("out_err_W4A4", "M6 layer-output rel. error, W4A4 NVFP4", False),
    ]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(10, 2.3 * len(metrics)), sharex="col")
    for j, (m, title) in enumerate(MODELS.items()):
        lin = display_kind(load(root, m, "linears"))
        lin = lin[lin.kind.isin(KIND_ORDER)]
        agg = lin.groupby(["kind", "layer"]).agg(
            {k: ("max" if k in ("absmax", "ch_max_over_median", "kurt_mean", "out_err_W4A4") else "min")
             for k, _, _ in metrics}
        ).reset_index()
        for i, (k, label, logy) in enumerate(metrics):
            ax = axes[i, j]
            for kind in KIND_ORDER:
                sub = agg[agg.kind == kind].sort_values("layer")
                if len(sub):
                    col, mk = KIND_STYLE[kind]
                    ax.plot(sub.layer, sub[k], color=col, marker=mk, markersize=3, label=KIND_LABEL.get(kind, kind))
            if logy:
                ax.set_yscale("log")
            ax.set_title(f"{title}: {label}" if i == 0 else label)
            if i == 0:
                ax.legend(ncol=3, fontsize=7, loc="upper left")
        axes[-1, j].set_xlabel("layer")
    fig.tight_layout()
    fig.savefig(out / "linear_inputs.png")
    plt.close(fig)


def fig_attention(root: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
    for j, (m, title) in enumerate(MODELS.items()):
        a = load(root, m, "attention")
        piv = a.pivot(index="layer", columns="head", values="attn_to_first")
        ax = axes[j]
        im = ax.imshow(piv.to_numpy(), aspect="auto", cmap=SEQ_CMAP, vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks(range(len(piv.index)), [str(i) for i in piv.index], fontsize=6.5)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.grid(False)
        ax.set_title(f"{title}: attention to first token (sink ratio {a.is_sink_head.mean():.2f})")
    fig.colorbar(im, ax=axes, shrink=0.8, label="mean attention to token 0")
    fig.savefig(out / "attention_sink.png")
    plt.close(fig)


def fig_weights(root: Path, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 4.8), sharex="col")
    for j, (m, title) in enumerate(MODELS.items()):
        w = display_kind(load(root, m, "weights"))
        w = w[w.kind.isin(KIND_ORDER)]
        agg = w.groupby(["kind", "layer"]).agg({"kurtosis": "max", "rel_err_nvfp4": "max"}).reset_index()
        for i, (k, label) in enumerate((("kurtosis", "M8 weight excess kurtosis (max in kind)"),
                                        ("rel_err_nvfp4", "M8 W-only NVFP4 rel. error"))):
            ax = axes[i, j]
            for kind in KIND_ORDER:
                sub = agg[agg.kind == kind].sort_values("layer")
                if len(sub):
                    col, mk = KIND_STYLE[kind]
                    ax.plot(sub.layer, sub[k], color=col, marker=mk, markersize=3, label=KIND_LABEL.get(kind, kind))
            ax.set_title(f"{title}: {label}" if i == 0 else label)
            if i == 0:
                ax.set_yscale("log")
                ax.legend(ncol=3, fontsize=7, loc="upper left")
        axes[-1, j].set_xlabel("layer")
    fig.tight_layout()
    fig.savefig(out / "weights.png")
    plt.close(fig)


def fig_peak_hist(root: Path, out: Path) -> None:
    """Distribution of per-token peak |u| at the residual reads with the strongest sink."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    bins = np.linspace(0, 32, 65)
    for j, (m, title) in enumerate(MODELS.items()):
        tv = np.load(root / m / "tokens.npz")
        r = load(root, m, "residual")
        r = r[(r.kind == "residual")].sort_values("depth")
        n = len(r)
        picks = [r.iloc[i]["name"] for i in (2, n // 2, n - 2)]
        ax = axes[j]
        for name, col in zip(picks, BLUE_RAMP[::-1]):
            peak, cat = tv[f"{name}/peak"], tv[f"{name}/cat"]
            ax.hist(peak[cat != 0], bins=bins, density=True, histtype="step", color=col, linewidth=1.5,
                    label=name.replace("model.layers.", "L"))
        ax.set_title(f"{title}: per-token peak |u_j| (non-first tokens)")
        ax.set_xlabel("max_j |u_j|")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(out / "peak_hist.png")
    plt.close(fig)


def fig_rtn(root: Path, out: Path) -> None:
    frames = []
    for m in MODELS:
        p = root / m / "rtn.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p).assign(model=m))
    if not frames:
        return
    df = pd.concat(frames)
    full = df[(df.kind == "all")]
    per_kind = df[(df.kind != "all") & (df.kind != "none")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.2, 1]})
    schemes = [s for s in ["W8A8", "W8A8-FP8", "W4A16", "W4A8", "W4A4", "A8", "A4", "A4-INT"] if s in set(full.scheme)]
    width = 0.38
    for i, m in enumerate(MODELS):
        sub = full[full.model == m].set_index("scheme").reindex(schemes)
        axes[0].bar(np.arange(len(schemes)) + (i - 0.5) * width, 100 * sub.dppl_rel, width=width * 0.94,
                    color=MODEL_COLOR[m], label=MODELS[m])
    axes[0].set_xticks(range(len(schemes)), schemes, fontsize=7.5)
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_ylabel("ΔPPL [%] (WikiText-2)")
    axes[0].set_title("RTN fake-quant, whole model")
    axes[0].legend()
    kinds = [k for k in KIND_ORDER if k in set(per_kind.kind)]
    labels = []
    xs = np.arange(len(kinds) * 2)
    for i, m in enumerate(MODELS):
        vals = []
        for k in kinds:
            for s in ("A4", "A4-INT"):
                v = per_kind[(per_kind.model == m) & (per_kind.kind == k) & (per_kind.scheme == s)].dppl_rel
                vals.append(100 * float(v.iloc[0]) if len(v) else np.nan)
        axes[1].bar(xs + (i - 0.5) * width, vals, width=width * 0.94, color=MODEL_COLOR[m], label=MODELS[m])
    for k in kinds:
        labels += [f"{k}\nNVFP4", f"{k}\nINT4"]
    axes[1].set_xticks(xs, labels, fontsize=6.5)
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].set_title("activation-only 4-bit, one module kind at a time")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "rtn.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/phase1"))
    ap.add_argument("--out", type=Path, default=Path("docs/reports/figs/phase1"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    setup_style()
    for fn in (fig_depth_profile, fig_block_outputs, fig_residual_sink, fig_sink_dims, fig_linear_inputs,
               fig_attention, fig_weights, fig_peak_hist, fig_rtn):
        fn(args.root, args.out)
        print("wrote", fn.__name__)
    for m in MODELS:
        mc = massive_counts(args.root, m)
        mc.to_csv(args.root / m / "massive_counts.csv", index=False)
        print(m, "massive-activation tokens (max over reads):",
              mc[["n_first", "n_delim", "n_other"]].max().to_dict(), "of", int(mc.n_first_total.iloc[0]), "seqs")


if __name__ == "__main__":
    main()
