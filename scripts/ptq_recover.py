"""Phase 3 P3: recover a weight-quantized model by training only a small bf16 part (the quantized Linears stay frozen).

L = KL(p_teacher ‖ p_student), the teacher being the original FP model (M0) in fp32 through the same bf16 autocast
path, as in Phase 2. Trainable sets:
  gates  - the retrofit gates (GatedNorm gn_down/gn_up, attention-gate ga_proj) + every RMSNorm weight
  norms  - every RMSNorm weight
  lora   - rank-r LoRA on the quantized Linears + every RMSNorm weight (merged into the weights when saving)
The result is written in the retrofit format (results/phase3/recover/<run>/final.pt) for scripts/eval_retrofit.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import nn

from outliers.losses import fused_kl
from outliers.models import load_model
from outliers.ptq import LoRAAdapters
from outliers.retrofit import load_retrofit, save_retrofit
from outliers.training import TokenStream, lr_factor

WANDB_PROJECT = "OutliersInLLM"
GATE_KEYS = ("gn_down", "gn_up", "ga_proj")


def trainable(model: nn.Module, which: str) -> dict[str, nn.Parameter]:
    out = {}
    for n, p in model.named_parameters():
        is_norm = n.endswith("norm.weight")
        is_gate = any(f".{k}." in n for k in GATE_KEYS)
        if is_norm or (which == "gates" and is_gate):
            out[n] = p
    return out


@torch.no_grad()
def heldout_kl(student: nn.Module, teacher: nn.Module, held: torch.Tensor) -> float:
    kls = []
    for s in range(0, held.shape[0], 4):
        ids = held[s : s + 4].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = student.model(input_ids=ids, use_cache=False).last_hidden_state
            ht = teacher.model(input_ids=ids, use_cache=False).last_hidden_state
        kls.append(float(fused_kl(hs, student.lm_head, ht, teacher.lm_head)))
    return float(np.mean(kls))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="results/phase3/<name>/quantized.pt")
    ap.add_argument("--train", choices=["gates", "norms", "lora"], required=True)
    ap.add_argument("--lora-rank", type=int, default=4)
    ap.add_argument("--run", required=True, help="output: results/phase3/recover/<run>")
    ap.add_argument("--tokens", type=float, default=50e6)
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (pilot) without changing the schedule")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--global-batch", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--probe-every", type=int, default=100)
    ap.add_argument("--probe-seqs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data", type=Path, default=Path("results/data/fineweb_edu"))
    ap.add_argument("--out-root", type=Path, default=Path("results/phase3/recover"))
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = args.out_root / args.run
    out.mkdir(parents=True, exist_ok=True)
    student, _, cfg, extra = load_retrofit(args.ckpt, dtype=torch.float32)
    base = extra["base"]
    teacher, _ = load_model(base, dtype=torch.float32)
    teacher.requires_grad_(False)
    student.requires_grad_(False)
    params = trainable(student, args.train)
    lora = LoRAAdapters(student, args.lora_rank) if args.train == "lora" else None
    train_params = list(params.values()) + (list(lora.parameters()) if lora is not None else [])
    for p in train_params:
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in train_params)
    student.train()
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.config.use_cache = False

    accum = args.global_batch // args.micro_batch
    total_steps = int(args.tokens // (args.global_batch * args.seq_len))
    warmup = max(1, int(args.warmup_frac * total_steps))
    last_step = min(total_steps, args.max_steps or total_steps)
    opt = torch.optim.AdamW(train_params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, fused=True)
    data = TokenStream(args.data / base / "train.npy", args.seq_len, args.seed)
    held = TokenStream(args.data / base / "heldout.npy", args.seq_len, 1).batch(0, args.probe_seqs)

    run_cfg = vars(args) | {"base": base, "retrofit": cfg.__dict__, "source": extra, "n_train_params": n_train,
                            "total_steps": total_steps, "warmup": warmup, "accum": accum}
    (out / "config.json").write_text(json.dumps(run_cfg, indent=2, default=str))
    run = None if args.no_wandb else wandb.init(project=WANDB_PROJECT, group="phase3-recover", name=args.run,
                                                config=run_cfg)
    log_f = (out / "metrics.jsonl").open("w")

    def log(step: int, m: dict[str, float]) -> None:
        log_f.write(json.dumps({"step": step} | m) + "\n")
        log_f.flush()
        if run is not None:
            run.log(m, step=step)

    print(f"training {n_train / 1e6:.3f}M parameters ({args.train}) for {last_step} steps", flush=True)
    log(0, {"heldout/kl": heldout_kl(student, teacher, held)})
    t_last = time.perf_counter()
    for step in range(last_step):
        f = lr_factor(step, total_steps, warmup, args.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = args.lr * f
        kl_sum = 0.0
        for micro in range(accum):
            ids = data.batch((step * accum + micro) * args.micro_batch, args.micro_batch).cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    th = teacher.model(input_ids=ids, use_cache=False).last_hidden_state
                h = student.model(input_ids=ids, use_cache=False).last_hidden_state
            loss = fused_kl(h, student.lm_head, th, teacher.lm_head)
            (loss / accum).backward()
            kl_sum += float(loss.detach()) / accum
        gnorm = float(torch.nn.utils.clip_grad_norm_(train_params, args.clip))
        opt.step()
        opt.zero_grad(set_to_none=True)
        now = time.perf_counter()
        tps = args.global_batch * args.seq_len / (now - t_last)
        t_last = now
        m = {"train/kl": kl_sum, "train/lr_factor": f, "train/grad_norm": gnorm, "train/tok_per_s": tps}
        done = step + 1
        if done % args.probe_every == 0 or done == last_step:
            student.eval()
            m["heldout/kl"] = heldout_kl(student, teacher, held)
            student.train()
        log(done, m)
        print(f"step {done}/{last_step} kl={kl_sum:.4f} gnorm={gnorm:.3f} tok/s={tps:.0f}"
              + (f" heldout_kl={m['heldout/kl']:.4f}" if "heldout/kl" in m else ""), flush=True)

    if lora is not None:
        lora.merge()
    student.to(torch.bfloat16)
    save_retrofit(student, cfg, base, out / "final.pt",
                  extra={"run": f"phase3/recover/{args.run}", "recover": args.train, "source_ckpt": str(args.ckpt),
                         "n_train_params": n_train})
    log_f.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
