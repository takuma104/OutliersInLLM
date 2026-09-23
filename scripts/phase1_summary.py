"""Markdown summary tables for the Phase 1 report, computed from results/phase1/<model>/*."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {"qwen3.5-0.8b": "Qwen3.5-0.8B-Base", "qwen3-0.6b": "Qwen3-0.6B-Base"}
KINDS = ["qkv", "in_proj_z", "o_proj", "out_proj", "gate_up", "down_proj"]


def md(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
    def fmt(v: object) -> str:
        if isinstance(v, (float, np.floating)):
            return "–" if np.isnan(v) else format(v, floatfmt)
        return str(v)

    def esc(v: str) -> str:
        return v.replace("|", "\\|")

    cols = [esc(str(c)) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(esc(fmt(v)) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def load(root: Path, m: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(root / m / f"{name}.parquet")


def display_kind(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["name"].str.contains("linear_attn.out_proj"), "kind"] = "out_proj"
    return df


def lm_table(root: Path) -> str:
    rows = []
    for m, title in MODELS.items():
        lm = json.loads((root / m / "lm.json").read_text())
        rows.append({"model": title, "WikiText-2 PPL": lm["wikitext2"]["ppl"], "WikiText-2 BPB": lm["wikitext2"]["bpb"],
                     "C4 probe PPL": lm["c4_probe"]["ppl"], "C4 probe BPB": lm["c4_probe"]["bpb"]})
    return md(pd.DataFrame(rows), ".4g")


def m1_table(root: Path) -> str:
    rows = []
    for m, title in MODELS.items():
        r = load(root, m, "residual")
        res, blk = r[r.kind == "residual"], r[r.kind == "block"]
        mc = pd.read_csv(root / m / "massive_counts.csv") if (root / m / "massive_counts.csv").exists() else None
        top = res.loc[res.top1.idxmax()]
        step_first = blk.loc[blk.max_first.idxmax()]
        rows.append(
            {
                "model": title,
                "max |h| first": res.max_first.max(),
                "max |h| delim": res.max_delim.max(),
                "max |h| other": res.max_other.max(),
                "median |h| (mid depth)": float(res.median_abs.iloc[len(res) // 2]),
                "top-1 dim": int(top.top1_dim),
                "MA tokens first/delim/other": "–" if mc is None else
                f"{int(mc.n_first.max())}/{int(mc.n_delim.max())}/{int(mc.n_other.max())}",
                "largest first-token block out": f"{step_first['name'].replace('model.layers.', 'L')} ({step_first.max_first:.3g})",
            }
        )
    return md(pd.DataFrame(rows))


def sink_table(root: Path) -> str:
    rows = []
    for m, title in MODELS.items():
        r = load(root, m, "residual")
        res = r[(r.kind == "residual") & (r["sub"] != "final")]
        norms = load(root, m, "norms")
        lam = {}
        for site in ("attn", "mlp"):
            vals = []
            for row in res[res["sub"] == site].itertuples():
                sub = norms[norms.norm == row.name].set_index("dim")["lambda_eff"]
                vals.append(abs(float(sub.loc[row.sink_dim])))
            lam[site] = np.median(vals)
        dims = res.sink_dim.value_counts()
        rows.append(
            {
                "model": title,
                "sink dim (#reads)": ", ".join(f"{d} ({c})" for d, c in dims.head(3).items()),
                "M2 score median [min–max]": f"{res.sink_score.median():.1f} [{res.sink_score.min():.1f}–{res.sink_score.max():.1f}]",
                "M2 argmax-hit median": res.sink_argmax_frac.median(),
                "#reads M2≥10": f"{int((res.sink_score >= 10).sum())}/{len(res)}",
                "M3 p50 median [max]": f"{res.peak_p50.median():.1f} [{res.peak_p50.max():.1f}]",
                "M3 p99 median": res.peak_p99.median(),
                "#reads M3 p50≥10": f"{int((res.peak_p50 >= 10).sum())}/{len(res)}",
                "e_top4 mean": res.e_top4_mean.median(),
                "M4 |λ| @sink, attn-norm": lam["attn"],
                "M4 |λ| @sink, mlp-norm": lam["mlp"],
            }
        )
    return md(pd.DataFrame(rows))


def linear_table(root: Path) -> str:
    frames = []
    for m, title in MODELS.items():
        lin = display_kind(load(root, m, "linears"))
        lin = lin[lin.kind.isin(KINDS)]
        g = lin.groupby("kind")
        t = pd.DataFrame(
            {
                "absmax (max)": g.absmax.max(),
                "absmax rest (median)": g.absmax_rest.median(),
                "ch max/med (median)": g.ch_max_over_median.median(),
                "ch RMS max/med (median)": g.ch_rms_max_over_median.median(),
                "#ch RMS>6×med (median)": g.n_ch_rms_over_6x.median(),
                "kurt (median)": g.kurt_mean.median(),
                "SQNR INT8 [dB]": g.sqnr_int8_tok.median(),
                "SQNR FP8": g.sqnr_fp8_tok.median(),
                "SQNR NVFP4": g.sqnr_nvfp4.median(),
                "SQNR INT4": g.sqnr_int4_tok.median(),
                "SQNR INT4 (min)": g.sqnr_int4_tok.min(),
                "err W8A8": g.out_err_W8A8.median(),
                "err W4A16": g.out_err_W4A16.median(),
                "err W4A4": g.out_err_W4A4.median(),
            }
        ).reindex([k for k in KINDS if k in g.groups])
        t.insert(0, "kind", t.index)
        t.insert(0, "model", title)
        frames.append(t)
    return md(pd.concat(frames, ignore_index=True))


def attention_table(root: Path) -> str:
    rows = []
    for m, title in MODELS.items():
        a = load(root, m, "attention")
        row = {
            "model": title,
            "#layers×heads": f"{a.layer.nunique()}×{a['head'].nunique()}",
            "sink ratio (>0.3)": a.is_sink_head.mean(),
            "attn→first mean": a.attn_to_first.mean(),
            "attn→first max": a.attn_to_first.max(),
            "v-norm first/rest (median)": a.v_norm_ratio.median(),
            "o-in norm first/rest (median)": float((a.o_norm_first / a.o_norm_rest).median()),
        }
        if "gate_rest" in a:
            row |= {"gate first": a.gate_first.mean(), "gate rest": a.gate_rest.mean(),
                    "gate<0.1 frac": a.gate_small_frac_rest.mean()}
        rows.append(row)
    return md(pd.DataFrame(rows))


def weight_table(root: Path) -> str:
    frames = []
    for m, title in MODELS.items():
        w = display_kind(load(root, m, "weights"))
        w = w[w.kind.isin(KINDS)]
        g = w.groupby("kind")
        t = pd.DataFrame(
            {
                "kurtosis median": g.kurtosis.median(),
                "kurtosis max": g.kurtosis.max(),
                "absmax/std max": g.absmax_over_std.max(),
                "in-ch norm max/med (median)": g.in_ch_norm_max_over_median.median(),
                "rel err INT8": g.rel_err_int8_ch.median(),
                "rel err INT4 ch": g.rel_err_int4_ch.median(),
                "rel err INT4 g128": g.rel_err_int4_g128.median(),
                "rel err NVFP4": g.rel_err_nvfp4.median(),
            }
        ).reindex([k for k in KINDS if k in g.groups])
        t.insert(0, "kind", t.index)
        t.insert(0, "model", title)
        frames.append(t)
    return md(pd.concat(frames, ignore_index=True))


def rtn_table(root: Path) -> str:
    frames = []
    for m, title in MODELS.items():
        p = root / m / "rtn.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p).assign(model=title))
    if not frames:
        return "(not run)"
    df = pd.concat(frames)
    df["label"] = np.where(df.kind.isin(["all", "none"]), df.scheme, df.scheme + " @" + df.kind)
    piv = df.pivot_table(index="label", columns="model", values="dppl_rel", sort=False) * 100
    ppl = df.pivot_table(index="label", columns="model", values="ppl", sort=False)
    out = pd.DataFrame({"setting": piv.index})
    for t in MODELS.values():
        if t in piv:
            out[f"{t} PPL"] = ppl[t].to_numpy()
            out[f"{t} ΔPPL %"] = piv[t].to_numpy()
    return md(out, ".4g")


def ablation_tables(root: Path) -> str:
    parts = []
    for m, title in MODELS.items():
        p = root / m / "ablate_sink.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["setting"] = np.where(df["mode"] == "clamp", "clamp τ=" + df.tau.astype(int).astype(str), df["mode"])
        allrow = df[df.scope == "all"][["setting", "ppl", "dppl_rel"]].assign(dppl_rel=lambda d: 100 * d.dppl_rel)
        single = df[~df.scope.isin(["all", "none"])]
        agg = single.groupby("setting", sort=False).dppl_rel.agg(["median", "max"]) * 100
        worst = single.loc[single.groupby("setting", sort=False).dppl_rel.idxmax()].set_index("setting").scope
        t = allrow.set_index("setting").join(agg).join(worst.rename("worst single read")).reset_index()
        t = t.rename(columns={"ppl": "PPL (all norms)", "dppl_rel": "ΔPPL % (all norms)",
                              "median": "single-read ΔPPL % median", "max": "single-read ΔPPL % max"})
        t["worst single read"] = t["worst single read"].str.replace("model.layers.", "L")
        parts.append(f"**{title}** (sink dims: {', '.join(map(str, sorted(single.sink_dim.unique())))})\n\n"
                     + md(t, ".4g"))
    return "\n\n".join(parts) if parts else "(not run)"


def step_up_table(root: Path) -> str:
    frames = []
    for m, title in MODELS.items():
        p = root / m / "step_up.parquet"
        if p.exists():
            s = pd.read_parquet(p)
            s = s.loc[s.nlargest(4, "uk_top1_over_median").index].sort_values("layer")
            frames.append(s.assign(model=title)[["model", "layer", "uk_top1_dim", "uk_top1_over_median",
                                                 "down_in_first_argmax_ch", "down_in_first_absmax",
                                                 "super_weight_row", "super_weight_over_std"]])
    return md(pd.concat(frames)) if frames else "(not run)"


def instruct_table(root: Path) -> str:
    """G: Base vs Instruct on the same probe."""
    rows = []
    for base, title in MODELS.items():
        for m in (base, f"{base}-instruct"):
            if not (root / m / "residual.parquet").exists():
                continue
            r = load(root, m, "residual")
            res = r[(r.kind == "residual") & (r["sub"] != "final")]
            lin = load(root, m, "linears")
            attn = load(root, m, "attention")
            norms = load(root, m, "norms")
            lam = []
            for row in res[res["sub"] == "mlp"].itertuples():
                lam.append(abs(float(norms[norms.norm == row.name].set_index("dim")["lambda_eff"].loc[row.sink_dim])))
            lm = json.loads((root / m / "lm.json").read_text())
            rows.append(
                {
                    "model": title.removesuffix("-Base") + (" Instruct" if m.endswith("instruct") else " Base"),
                    "WikiText-2 PPL": lm["wikitext2"]["ppl"],
                    "max |h| (final norm 除く)": res.top1.max(),
                    "sink dim": int(res.sink_dim.mode().iloc[0]),
                    "M2 median": res.sink_score.median(),
                    "M3 p50 median": res.peak_p50.median(),
                    "|λ| @sink mlp-norm": float(np.median(lam)),
                    "sink ratio": attn.is_sink_head.mean(),
                    "down_proj absmax": lin[lin.kind == "down_proj"].absmax.max(),
                    "SQNR INT4 median": lin.sqnr_int4_tok.median(),
                }
            )
    return md(pd.DataFrame(rows)) if rows else "(not run)"


def input_dependence_table(root: Path) -> str:
    p = root / "input_dependence.parquet"
    if not p.exists():
        return "(not run)"
    df = pd.read_parquet(p)
    df = df[df["sub"] != "final"]
    rows = []
    for (m, dom), g in df.groupby(["model", "domain"], sort=False):
        mode = int(g.sink_dim.mode().iloc[0])
        rows.append({"model": MODELS[m], "domain": dom, "sink dim": mode, "share of reads": float((g.sink_dim == mode).mean()),
                     "M2 median": g.sink_score.median(), "argmax-hit median": g.sink_argmax_frac.median(),
                     "M3 p50 median": g.peak_p50.median(), "max |h| first": g.max_first.max()})
    return md(pd.DataFrame(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/phase1"))
    args = ap.parse_args()
    sections = {
        "LM quality": lm_table,
        "M1 massive activations": m1_table,
        "M2-M4 residual sink": sink_table,
        "M5/M6 Linear inputs (median over layers)": linear_table,
        "M7 attention": attention_table,
        "M8 weights (median over layers)": weight_table,
        "F RTN fake-quant ΔPPL (WikiText-2)": rtn_table,
        "E sink ablation": ablation_tables,
        "H step-up / super weights": step_up_table,
        "G Base vs Instruct": instruct_table,
        "Input dependence (en / ja / code)": input_dependence_table,
    }
    for title, fn in sections.items():
        print(f"### {title}\n\n{fn(args.root)}\n")


if __name__ == "__main__":
    main()
