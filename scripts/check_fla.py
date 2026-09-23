"""Phase 1 Day 1: check that fla GDN kernels run on this GPU and match the torch reference (plan §3.1).

1. kernel level: fla chunk / fused_recurrent gated delta rule vs transformers' torch reference (fp32)
2. model level: Qwen3.5 logits with fla vs torch reference, both compared to an fp32 torch-reference run
3. throughput: forward and forward+backward tokens/s with fla
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers.models.qwen3_5.modeling_qwen3_5 as mq
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

from outliers.data import build_probe_set, c4_validation_texts, select_probe_docs
from outliers.models import load_model

TORCH_CHUNK: Callable = mq.torch_chunk_gated_delta_rule.__wrapped__
TORCH_RECURRENT: Callable = mq.torch_recurrent_gated_delta_rule.__wrapped__


@contextlib.contextmanager
def torch_gdn_reference() -> Iterator[None]:
    """Route Qwen3.5 GDN through the pure-torch reference implementation."""
    saved = (mq.torch_chunk_gated_delta_rule, mq.torch_recurrent_gated_delta_rule)
    mq.torch_chunk_gated_delta_rule, mq.torch_recurrent_gated_delta_rule = TORCH_CHUNK, TORCH_RECURRENT
    try:
        yield
    finally:
        mq.torch_chunk_gated_delta_rule, mq.torch_recurrent_gated_delta_rule = saved


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm())


def kernel_check(seq_len: int, device: str = "cuda") -> dict[str, float]:
    torch.manual_seed(0)
    b, h, dk, dv = 2, 16, 128, 128
    q = torch.randn(b, seq_len, h, dk, device=device, dtype=torch.bfloat16)
    k = torch.randn(b, seq_len, h, dk, device=device, dtype=torch.bfloat16)
    v = torch.randn(b, seq_len, h, dv, device=device, dtype=torch.bfloat16)
    g = -F.softplus(torch.randn(b, seq_len, h, device=device)) * 0.5
    beta = torch.rand(b, seq_len, h, device=device, dtype=torch.bfloat16)

    ref, ref_state = TORCH_CHUNK(q.float(), k.float(), v.float(), g, beta.float(), output_final_state=True,
                                 use_qk_l2norm_in_kernel=True)
    out_c, st_c = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, output_final_state=True,
                                         use_qk_l2norm_in_kernel=True)
    res = {
        "chunk_out_rel_err": rel_err(out_c, ref),
        "chunk_state_rel_err": rel_err(st_c, ref_state),
    }
    # recurrent kernel (decode path) on a short prefix
    t = 256
    ref_r, _ = TORCH_RECURRENT(q[:, :t].float(), k[:, :t].float(), v[:, :t].float(), g[:, :t], beta[:, :t].float(),
                               use_qk_l2norm_in_kernel=True)
    out_r, _ = fused_recurrent_gated_delta_rule(q[:, :t], k[:, :t], v[:, :t], g=g[:, :t], beta=beta[:, :t],
                                                use_qk_l2norm_in_kernel=True)
    res["recurrent_out_rel_err"] = rel_err(out_r, ref_r)
    res["torch_chunk_vs_recurrent_rel_err"] = rel_err(ref[:, :t], ref_r)

    # backward through the chunk kernel
    qg, kg, vg = (x.clone().requires_grad_() for x in (q, k, v))
    out, _ = chunk_gated_delta_rule(qg, kg, vg, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    out.float().square().mean().backward()
    qf, kf, vf = (x.float().clone().requires_grad_() for x in (q, k, v))
    out_ref, _ = TORCH_CHUNK(qf, kf, vf, g, beta.float(), use_qk_l2norm_in_kernel=True)
    out_ref.square().mean().backward()
    res["chunk_grad_q_rel_err"] = rel_err(qg.grad, qf.grad)
    res["chunk_grad_v_rel_err"] = rel_err(vg.grad, vf.grad)
    return res


@torch.no_grad()
def logits_of(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    return model(input_ids=ids).logits.float()


def kl_mean(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """mean over tokens of KL(p || q)."""
    lp, lq = F.log_softmax(p_logits, -1), F.log_softmax(q_logits, -1)
    return float((lp.exp() * (lp - lq)).sum(-1).mean())


def ce(logits: torch.Tensor, ids: torch.Tensor) -> float:
    return float(F.cross_entropy(logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten()))


def model_check(n_seq: int, seq_len: int) -> dict[str, float]:
    model, tok = load_model("qwen3.5-0.8b")
    model32, _ = load_model("qwen3.5-0.8b", dtype=torch.float32)
    texts = c4_validation_texts()
    idx = select_probe_docs(texts, [tok], n_docs=n_seq, seq_len=seq_len, seed=1234)
    ids = build_probe_set(tok, texts, idx, seq_len).input_ids.cuda()

    acc: dict[str, float] = {}
    for i in range(n_seq):
        x = ids[i : i + 1]
        with torch_gdn_reference():
            c = logits_of(model32, x)
            b = logits_of(model, x)
        a = logits_of(model, x)
        per = {
            "kl_fp32ref_vs_bf16_fla": kl_mean(c, a),
            "kl_fp32ref_vs_bf16_torch": kl_mean(c, b),
            "kl_bf16_torch_vs_bf16_fla": kl_mean(b, a),
            "ce_fp32_torch": ce(c, x),
            "ce_bf16_fla": ce(a, x),
            "ce_bf16_torch": ce(b, x),
            "argmax_agree_fla_vs_fp32": float((a.argmax(-1) == c.argmax(-1)).float().mean()),
            "argmax_agree_torch_vs_fp32": float((b.argmax(-1) == c.argmax(-1)).float().mean()),
        }
        for k, v in per.items():
            acc[k] = acc.get(k, 0.0) + v / n_seq
        del a, b, c
    del model, model32
    torch.cuda.empty_cache()
    return acc


def throughput(batch: int, seq_len: int, steps: int = 5) -> dict[str, float]:
    model, _ = load_model("qwen3.5-0.8b")
    ids = torch.randint(0, 200_000, (batch, seq_len), device="cuda")
    res: dict[str, float] = {}

    def bench(fn: Callable[[], None]) -> float:
        fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            fn()
        torch.cuda.synchronize()
        return batch * seq_len * steps / (time.perf_counter() - t0)

    with torch.no_grad():
        res["fwd_tok_s_fla"] = bench(lambda: model.model(input_ids=ids))
        with torch_gdn_reference():
            res["fwd_tok_s_torch"] = bench(lambda: model.model(input_ids=ids))

    # training-style step: bf16 weights with grad, hidden-state loss (avoids 248K-vocab logits)
    model.train()
    model.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    torch.cuda.reset_peak_memory_stats()

    def fwd_bwd() -> None:
        h = model.model(input_ids=ids).last_hidden_state
        h.float().square().mean().backward()
        model.zero_grad(set_to_none=True)

    res["fwd_bwd_tok_s_fla_gc"] = bench(fwd_bwd)
    res["fwd_bwd_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/phase1/check_fla.json"))
    ap.add_argument("--n-seq", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=2048)
    args = ap.parse_args()

    print(f"device: {torch.cuda.get_device_name()} cc={torch.cuda.get_device_capability()}")
    report: dict[str, dict[str, float]] = {}
    report["kernel"] = kernel_check(args.seq_len)
    print(json.dumps(report["kernel"], indent=2))
    report["model"] = model_check(args.n_seq, args.seq_len)
    print(json.dumps(report["model"], indent=2))
    report["throughput"] = throughput(batch=8, seq_len=args.seq_len)
    print(json.dumps(report["throughput"], indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
