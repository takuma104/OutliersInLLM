"""Phase 2 retrofit fine-tuning: L = KL(p_orig ‖ p_θ) + λ·R (Track A) or CE (Track B).

The teacher is the original model in fp32 run through the same bf16 autocast path as the student, so an
identity-initialized student has KL = 0 exactly at step 0. Probe metrics (outliers, gates, held-out KL) are logged
every ``--probe-every`` steps to <out>/metrics.jsonl and wandb.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import wandb
from torch import nn

from outliers.data import build_probe_set, c4_validation_texts, load_probe_doc_indices
from outliers.hooks import OutlierProbe
from outliers.losses import OutlierRegularizer, fused_ce, fused_kl
from outliers.models import load_model
from outliers.retrofit import RetrofitConfig, apply_retrofit, attn_gates, gated_norms, save_retrofit

WANDB_PROJECT = "OutliersInLLM"
PROBE_KINDS = ("qkv", "in_proj_z", "o_proj", "gate_up", "down_proj")


class TokenStream:
    """Packed token stream cut into ``seq_len`` sequences, visited in a fixed seeded order."""

    def __init__(self, path: Path, seq_len: int, seed: int) -> None:
        self.tokens = np.load(path, mmap_mode="r")
        self.seq_len = seq_len
        self.n_seq = len(self.tokens) // seq_len
        self.order = np.random.default_rng(seed).permutation(self.n_seq)

    def batch(self, start: int, count: int) -> torch.Tensor:
        rows = [self.order[(start + i) % self.n_seq] for i in range(count)]
        arr = np.stack([self.tokens[r * self.seq_len : (r + 1) * self.seq_len] for r in rows]).astype(np.int64)
        return torch.from_numpy(arr)


def param_groups(model: nn.Module, new: set[str], lr: float, lr_new: float, wd: float,
                 train_base: bool) -> list[dict[str, Any]]:
    decay, no_decay, fresh = [], [], []
    for n, p in model.named_parameters():
        if n in new:
            fresh.append(p)
        elif not train_base:
            p.requires_grad_(False)
        elif p.ndim == 2 and "embed_tokens" not in n:
            decay.append(p)  # Linear weights
        else:
            no_decay.append(p)  # norms, embeddings (tied lm_head), conv1d, A_log, dt_bias
    groups = [{"params": fresh, "lr": lr_new, "weight_decay": 0.0, "name": "new"}]
    if train_base:
        groups += [{"params": decay, "lr": lr, "weight_decay": wd, "name": "decay"},
                   {"params": no_decay, "lr": lr, "weight_decay": 0.0, "name": "no_decay"}]
    return [g for g in groups if g["params"]]


def lr_factor(step: int, total: int, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def gate_stats(model: nn.Module, ids: torch.Tensor) -> dict[str, float]:
    """GatedNorm / attention-gate statistics on one batch (plan §5.5 (ii))."""
    out: dict[str, float] = {}
    gns = gated_norms(model)
    ags = attn_gates(model)
    if not gns and not ags:
        return out
    acc: dict[str, list[float]] = {"gn_mean": [], "gn_min_dim": [], "gn_frac_lt_half": [], "gn_corr": [],
                                   "ga_mean": [], "ga_frac_lt_0.1": []}
    handles = []
    for mod in gns.values():

        def hook(m: nn.Module, args: tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
            y = m.norm(args[0]).float()
            g = m.gate(y.to(args[0].dtype)).float()
            acc["gn_mean"].append(float(g.mean()))
            acc["gn_min_dim"].append(float(g.mean((0, 1)).min()))
            acc["gn_frac_lt_half"].append(float((g < 0.5).float().mean()))
            ya, gm = y.abs().mean((0, 1)), g.mean((0, 1))  # per-dim |y| vs per-dim gate
            acc["gn_corr"].append(float(torch.corrcoef(torch.stack([ya, gm]))[0, 1]))

        handles.append(mod.register_forward_hook(hook))
    for mod in ags.values():

        def ga_hook(_m: nn.Module, _a: tuple[Any, ...], g: torch.Tensor) -> None:
            acc["ga_mean"].append(float(g.float().mean()))
            acc["ga_frac_lt_0.1"].append(float((g < 0.1).float().mean()))

        handles.append(mod.register_forward_hook(ga_hook))
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.model(input_ids=ids.cuda(), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    for k, v in acc.items():
        if v:
            out[f"gate/{k}"] = float(np.mean(v)) if k != "gn_min_dim" else float(np.min(v))
    return out


def zbias_stats(model: nn.Module) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for n, p in model.named_parameters():
        if n.endswith("zbias"):
            kind = n.rsplit(".", 2)[-2]
            out.setdefault(kind, []).append(float(p.detach().float().abs().mean()))
    return {f"bias/{k}_absmean": float(np.mean(v)) for k, v in out.items()}


@torch.no_grad()
def probe(student: nn.Module, teacher: nn.Module, probe_ids: torch.Tensor, probe_cats: torch.Tensor,
          held: torch.Tensor, out_dir: Path, step: int) -> dict[str, float]:
    student.eval()
    m: dict[str, float] = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with OutlierProbe(student, blocks=False, quant_error=False) as p:
            p.run(probe_ids, probe_cats, batch_size=4)
            res = p.results()
    r = res.residual[res.residual["sub"] != "final"]
    m |= {
        "probe/peak_p50_median": float(r.peak_p50.median()), "probe/peak_p50_max": float(r.peak_p50.max()),
        "probe/peak_p99_median": float(r.peak_p99.median()), "probe/peak_p99_max": float(r.peak_p99.max()),
        "probe/sink_score_median": float(r.sink_score.median()), "probe/sink_score_max": float(r.sink_score.max()),
        "probe/e_top4_median": float(r.e_top4_mean.median()),
        "probe/max_first": float(res.residual.max_first.max()), "probe/max_other": float(res.residual.max_other.max()),
        "probe/sink_argmax_median": float(r.sink_argmax_frac.median()),
    }
    lin = res.linears
    lin.loc[lin["name"].str.contains("linear_attn.out_proj"), "kind"] = "out_proj"
    for kind in (*PROBE_KINDS, "out_proj"):
        sub = lin[lin.kind == kind]
        if len(sub):
            m[f"probe/{kind}_absmax_max"] = float(sub.absmax.max())
            m[f"probe/{kind}_chrms_median"] = float(sub.ch_rms_max_over_median.median())
            m[f"probe/{kind}_kurt_median"] = float(sub.kurt_mean.median())
    probe_dir = out_dir / "probe"
    probe_dir.mkdir(exist_ok=True)
    res.residual.to_parquet(probe_dir / f"{step:05d}_residual.parquet")
    lin.to_parquet(probe_dir / f"{step:05d}_linears.parquet")

    kls, ces = [], []
    for s in range(0, held.shape[0], 4):
        ids = held[s : s + 4].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = student.model(input_ids=ids, use_cache=False).last_hidden_state
            ht = teacher.model(input_ids=ids, use_cache=False).last_hidden_state
        kls.append(float(fused_kl(hs, student.lm_head, ht, teacher.lm_head)))
        ces.append(float(fused_ce(hs[:, :-1], student.lm_head, ids[:, 1:])))
    m["heldout/kl"] = float(np.mean(kls))
    m["heldout/ce"] = float(np.mean(ces))
    m |= gate_stats(student, probe_ids[:4])
    m |= zbias_stats(student)
    student.train()
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True, help="run name (results/phase2/<run>)")
    ap.add_argument("--group", default="phase2")
    ap.add_argument("--loss", choices=["kl", "ce"], default="kl")
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=8.0)
    ap.add_argument("--first-weight", type=float, default=1.0, help="weight of position 0 in R (1 = token mean)")
    ap.add_argument("--gated-norm", action="store_true")
    ap.add_argument("--gn-rank", type=int, default=16)
    ap.add_argument("--linear-bias", action="store_true")
    ap.add_argument("--attn-gate", choices=["headwise", "elementwise"], default=None)
    ap.add_argument("--new-only", action="store_true", help="freeze original weights (A3)")
    ap.add_argument("--tokens", type=float, default=200e6)
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (pilot) without changing the schedule")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--global-batch", type=int, default=128)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr-new", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--probe-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--probe-seqs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", type=Path, default=Path("results/data/fineweb_edu"))
    ap.add_argument("--out-root", type=Path, default=Path("results/phase2"))
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = args.out_root / args.run
    out.mkdir(parents=True, exist_ok=True)
    cfg = RetrofitConfig(gated_norm=args.gated_norm, gated_norm_rank=args.gn_rank, linear_bias=args.linear_bias,
                         attn_gate=args.attn_gate)
    teacher, tok = load_model(args.model, dtype=torch.float32)
    student, _ = load_model(args.model, dtype=torch.float32)
    new = set(apply_retrofit(student, cfg))
    student.train()
    student.requires_grad_(True)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.config.use_cache = False
    reg = OutlierRegularizer(student, tau=args.tau, first_weight=args.first_weight)

    accum = args.global_batch // args.micro_batch
    total_steps = int(args.tokens // (args.global_batch * args.seq_len))
    warmup = max(1, int(args.warmup_frac * total_steps))
    last_step = min(total_steps, args.max_steps or total_steps)
    groups = param_groups(student, new, args.lr, args.lr_new, args.wd, train_base=not args.new_only)
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, fused=True)

    data = TokenStream(args.data / args.model / "train.npy", args.seq_len, args.seed)
    held = TokenStream(args.data / args.model / "heldout.npy", args.seq_len, args.seed + 1).batch(0, args.probe_seqs)
    probe_set = build_probe_set(tok, c4_validation_texts(), load_probe_doc_indices()[: args.probe_seqs], args.seq_len)

    start = 0
    state_path = out / "last_state.pt"
    if state_path.exists():
        st = torch.load(state_path, map_location="cuda", weights_only=False)
        student.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start = st["step"]
        print(f"resumed from step {start}", flush=True)

    run_cfg = vars(args) | {"retrofit": dataclasses.asdict(cfg), "total_steps": total_steps, "warmup": warmup,
                            "accum": accum, "n_new_params": sum(p.numel() for n, p in student.named_parameters()
                                                                if n in new)}
    (out / "config.json").write_text(json.dumps(run_cfg, indent=2, default=str))
    run = None if args.no_wandb else wandb.init(project=WANDB_PROJECT, group=args.group, name=args.run,
                                                config=run_cfg, resume="allow", id=args.run.replace("/", "-"))
    log_f = (out / "metrics.jsonl").open("a")

    def log(step: int, m: dict[str, float]) -> None:
        log_f.write(json.dumps({"step": step} | m) + "\n")
        log_f.flush()
        if run is not None:
            run.log(m, step=step)

    if start == 0:
        log(0, probe(student, teacher, probe_set.input_ids, probe_set.categories, held, out, 0))

    t_last = time.perf_counter()
    for step in range(start, last_step):
        f = lr_factor(step, total_steps, warmup, args.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        kl_sum = r_sum = 0.0
        per_norm_acc: dict[str, float] = {}
        for micro in range(accum):
            ids = data.batch((step * accum + micro) * args.micro_batch, args.micro_batch).cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.loss == "kl":
                    with torch.no_grad():
                        th = teacher.model(input_ids=ids, use_cache=False).last_hidden_state
                with reg.record():
                    h = student.model(input_ids=ids, use_cache=False).last_hidden_state
            r, per_norm = reg.collect()
            if args.loss == "kl":
                main_loss = fused_kl(h, student.lm_head, th, teacher.lm_head)
            else:
                main_loss = fused_ce(h[:, :-1], student.lm_head, ids[:, 1:])
            loss = (main_loss + args.lam * r) / accum
            loss.backward()
            kl_sum += float(main_loss.detach()) / accum
            r_sum += float(r.detach()) / accum
            for k, v in per_norm.items():
                per_norm_acc[k] = per_norm_acc.get(k, 0.0) + v / accum
        gnorm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip))
        opt.step()
        opt.zero_grad(set_to_none=True)
        now = time.perf_counter()
        tps = args.global_batch * args.seq_len / (now - t_last)
        t_last = now
        m = {"train/" + ("kl" if args.loss == "kl" else "ce"): kl_sum, "train/R": r_sum, "train/lr_factor": f,
             "train/grad_norm": gnorm, "train/tok_per_s": tps,
             "train/R_max_norm": max(per_norm_acc.values())}
        done = step + 1
        if done % args.probe_every == 0 or done == last_step:
            reg.deactivate()
            m |= probe(student, teacher, probe_set.input_ids, probe_set.categories, held, out, done)
            pd.Series(per_norm_acc).to_frame("R").to_parquet(out / "probe" / f"{done:05d}_R_per_norm.parquet")
        log(done, m)
        print(f"step {done}/{last_step} kl={kl_sum:.4f} R={r_sum:.3f} gnorm={gnorm:.3f} tok/s={tps:.0f}", flush=True)
        if done % args.save_every == 0 and done < last_step:
            torch.save({"model": student.state_dict(), "opt": opt.state_dict(), "step": done}, state_path)

    reg.remove()
    student.to(torch.bfloat16)
    save_retrofit(student, cfg, args.model, out / "final.pt", extra={"run": args.run, "steps": last_step})
    state_path.unlink(missing_ok=True)
    log_f.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
