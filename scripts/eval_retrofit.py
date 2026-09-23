"""Phase 2 evaluation of one checkpoint (or the original model): LM quality, outliers, RTN quantization, gate usage,
zero-shot tasks (plan §5.5). Writes results/phase2/<run>/eval/ and logs a wandb run "<run>-eval".
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from torch import nn

from outliers.data import (
    build_probe_set,
    c4_validation_texts,
    chunk_tokens,
    load_probe_doc_indices,
    token_byte_lengths,
    wikitext2_test_text,
)
from outliers.evaluate import evaluate_lm
from outliers.hooks import OutlierProbe
from outliers.losses import fused_kl
from outliers.models import NON_QUANT_KINDS, iter_linears, load_model, set_attn_impl
from outliers.quant import SCHEMES, fake_quantize
from outliers.retrofit import RetrofitConfig, attn_gates, gated_norms, load_retrofit
from outliers.stats import norm_weight_table, weight_stats_table

WANDB_PROJECT = "OutliersInLLM"
RTN_SCHEMES = ["W8A8", "W4A16", "W4A4", "A4", "A4-INT"]
LM_EVAL_TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "lambada_openai"]


def log(msg: str, t0: float) -> None:
    print(f"[{time.perf_counter() - t0:7.1f}s] {msg}", flush=True)


@torch.no_grad()
def heldout_kl(model: nn.Module, teacher: nn.Module, ids: torch.Tensor, batch: int = 4) -> float:
    vals = []
    for s in range(0, ids.shape[0], batch):
        x = ids[s : s + batch].cuda()
        hs = model.model(input_ids=x, use_cache=False).last_hidden_state
        ht = teacher.model(input_ids=x, use_cache=False).last_hidden_state
        vals.append(float(fused_kl(hs, model.lm_head, ht, teacher.lm_head)))
    return float(np.mean(vals))


@contextlib.contextmanager
def neutralize(model: nn.Module, what: str) -> Iterator[None]:
    """Temporarily undo one retrofit: GatedNorm gates -> 1, attention gates -> 1, zero biases -> 0."""
    saved: list[tuple[torch.Tensor, torch.Tensor]] = []
    if what == "gatednorm":
        params = [m.gn_up.weight for m in gated_norms(model).values()]
    elif what == "attn_gate":
        params = [p for m in attn_gates(model).values() for p in m.ga_proj.parameters()]
    elif what == "bias":
        params = [p for n, p in model.named_parameters() if n.endswith("zbias")]
    else:
        raise ValueError(what)
    for p in params:
        saved.append((p, p.data.clone()))
        p.data.zero_()
    try:
        yield
    finally:
        for p, d in saved:
            p.data.copy_(d)


def run_lm_eval(model: nn.Module, tok, limit: int | None, batch_size: int = 32) -> dict[str, float]:  # noqa: ANN001
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size, max_length=2048)
    res = simple_evaluate(model=lm, tasks=LM_EVAL_TASKS, num_fewshot=0, limit=limit, log_samples=False)
    out, main_scores = {}, []
    for task, r in res["results"].items():
        for key in ("acc_norm,none", "acc,none", "perplexity,none"):
            if key in r:
                out[f"lmeval/{task}/{key.split(',')[0]}"] = float(r[key])
        main_scores.append(float(r.get("acc_norm,none", r.get("acc,none", float("nan")))))
    out["lmeval/avg_acc"] = float(np.nanmean(main_scores))  # acc_norm where reported, else acc
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", type=Path, help="results/phase2/<run>/final.pt")
    g.add_argument("--base", help="evaluate the original model (e.g. qwen3.5-0.8b)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--probe-docs", type=int, default=128)
    ap.add_argument("--heldout-seqs", type=int, default=64)
    ap.add_argument("--lm-batch", type=int, default=8, help="batch size for PPL evaluation")
    ap.add_argument("--no-lm-eval", action="store_true")
    ap.add_argument("--lm-eval-limit", type=int, default=None)
    ap.add_argument("--lm-eval-batch", type=int, default=32)
    ap.add_argument("--lm-eval-only", action="store_true", help="only run lm-eval and merge into summary.json")
    ap.add_argument("--attention", action="store_true", help="M7 attention-sink probe (eager attention, 32 docs)")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--data", type=Path, default=Path("results/data/fineweb_edu"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.ckpt is not None:
        model, tok, cfg, extra = load_retrofit(args.ckpt)
        base, name = extra["base"], extra.get("run", args.ckpt.parent.name)
        out = args.out or args.ckpt.parent / "eval"
    else:
        model, tok = load_model(args.base)
        cfg, base, name = RetrofitConfig(), args.base, f"base-{args.base}"
        out = args.out or Path("results/phase2") / name / "eval"
    out.mkdir(parents=True, exist_ok=True)
    if args.lm_eval_only:
        summary = json.loads((out / "summary.json").read_text())
        summary |= run_lm_eval(model, tok, args.lm_eval_limit, args.lm_eval_batch)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        log(f"lm-eval avg acc {summary['lmeval/avg_acc']:.4f}", t0)
        return
    teacher, _ = load_model(base)
    summary: dict[str, float | str] = {"run": name, "base": base, "retrofit": json.dumps(cfg.__dict__)}
    log(f"loaded {name} ({cfg})", t0)

    bl = token_byte_lengths(tok)
    wt = chunk_tokens(wikitext2_test_text(), tok, 2048)
    probe = build_probe_set(tok, c4_validation_texts(), load_probe_doc_indices()[: args.probe_docs], 2048)
    held = torch.from_numpy(np.load(args.data / base / "heldout.npy", mmap_mode="r")[: args.heldout_seqs * 2048]
                            .astype(np.int64)).view(-1, 2048)
    lm = {"wikitext2": evaluate_lm(model, wt, bl, args.lm_batch),
          "c4_probe": evaluate_lm(model, probe.input_ids, bl, args.lm_batch)}
    for k, v in lm.items():
        summary |= {f"{k}/ppl": v["ppl"], f"{k}/bpb": v["bpb"]}
    summary["heldout/kl"] = heldout_kl(model, teacher, held)
    log(f"LM: wt2 ppl {lm['wikitext2']['ppl']:.3f}, held-out KL {summary['heldout/kl']:.4f}", t0)

    # gate usage: undo each retrofit and measure the KL increase
    for what, active in (("gatednorm", cfg.gated_norm), ("attn_gate", cfg.attn_gate is not None),
                         ("bias", cfg.linear_bias)):
        if active:
            with neutralize(model, what):
                summary[f"neutralize_{what}/heldout_kl"] = heldout_kl(model, teacher, held)
                summary[f"neutralize_{what}/wikitext2_ppl"] = evaluate_lm(model, wt, bl, args.lm_batch)["ppl"]
    log("gate usage done", t0)

    # outliers (Phase 1 M1-M8 on the C4 probe)
    norm_weight_table(model).to_parquet(out / "norms.parquet")
    weight_stats_table(model).to_parquet(out / "weights.parquet")
    with OutlierProbe(model) as p:
        p.run(probe.input_ids, probe.categories, batch_size=4)
        res = p.results()
    res.residual.to_parquet(out / "residual.parquet")
    res.residual_dims.to_parquet(out / "residual_dims.parquet")
    res.linears.to_parquet(out / "linears.parquet")
    r = res.residual[(res.residual.kind == "residual") & (res.residual["sub"] != "final")]
    summary |= {
        "outlier/max_first": float(r.max_first.max()), "outlier/max_other": float(r.max_other.max()),
        "outlier/sink_score_median": float(r.sink_score.median()), "outlier/sink_score_max": float(r.sink_score.max()),
        "outlier/peak_p50_median": float(r.peak_p50.median()), "outlier/peak_p50_max": float(r.peak_p50.max()),
        "outlier/peak_p99_median": float(r.peak_p99.median()), "outlier/peak_p99_max": float(r.peak_p99.max()),
        "outlier/e_top4_median": float(r.e_top4_mean.median()),
    }
    lin = res.linears.copy()
    lin.loc[lin["name"].str.contains("linear_attn.out_proj"), "kind"] = "out_proj"
    for kind, sub in lin[~lin.kind.isin(NON_QUANT_KINDS)].groupby("kind"):
        summary |= {f"linear/{kind}/absmax_max": float(sub.absmax.max()),
                    f"linear/{kind}/chrms_median": float(sub.ch_rms_max_over_median.median()),
                    f"linear/{kind}/sqnr_nvfp4_median": float(sub.sqnr_nvfp4.median()),
                    f"linear/{kind}/sqnr_int4_median": float(sub.sqnr_int4_tok.median()),
                    f"linear/{kind}/out_err_W4A4_median": float(sub.out_err_W4A4.median())}
    log("outlier probe done", t0)

    if args.attention:
        set_attn_impl(model, "eager")
        with OutlierProbe(model, residual=False, blocks=False, linears=False, attention=True) as p:
            p.run(probe.input_ids[:32], probe.categories[:32], batch_size=1)
            attn = p.results().attention
        set_attn_impl(model, "sdpa")
        attn.to_parquet(out / "attention.parquet")
        summary |= {"attn/sink_ratio": float(attn.is_sink_head.mean()),
                    "attn/to_first_mean": float(attn.attn_to_first.mean()),
                    "attn/v_norm_ratio_median": float(attn.v_norm_ratio.median())}
        log(f"attention probe done (sink ratio {summary['attn/sink_ratio']:.2f})", t0)

    # RTN fake quantization
    base_ppl = lm["wikitext2"]["ppl"]
    rtn_rows = []
    kinds = sorted({li.kind for li in iter_linears(model)} - NON_QUANT_KINDS)
    for scheme in RTN_SCHEMES:
        with fake_quantize(model, SCHEMES[scheme]):
            ppl = evaluate_lm(model, wt, bl, args.lm_batch)["ppl"]
        rtn_rows.append({"scheme": scheme, "kind": "all", "ppl": ppl})
    for kind in kinds:
        with fake_quantize(model, SCHEMES["A4-INT"], kinds=[kind]):
            rtn_rows.append({"scheme": "A4-INT", "kind": kind, "ppl": evaluate_lm(model, wt, bl, args.lm_batch)["ppl"]})
    rtn = pd.DataFrame(rtn_rows).assign(dppl_rel=lambda d: d.ppl / base_ppl - 1)
    rtn.to_parquet(out / "rtn.parquet")
    for row in rtn.itertuples():
        summary[f"rtn/{row.scheme}" + ("" if row.kind == "all" else f"@{row.kind}")] = float(row.dppl_rel)
    log("RTN done", t0)

    if not args.no_lm_eval:
        summary |= run_lm_eval(model, tok, args.lm_eval_limit, args.lm_eval_batch)
        log(f"lm-eval avg acc {summary['lmeval/avg_acc']:.4f}", t0)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if not args.no_wandb:
        run = wandb.init(project=WANDB_PROJECT, group="phase2-eval", job_type="eval", name=f"{name}-eval",
                         config={"run": name, "base": base, "retrofit": cfg.__dict__})
        run.summary.update({k: v for k, v in summary.items() if isinstance(v, float)})
        run.log({"rtn": wandb.Table(dataframe=rtn)})
        run.finish()
    log("finished", t0)


if __name__ == "__main__":
    main()
