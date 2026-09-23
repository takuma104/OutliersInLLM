"""Retrofits must be exact identities at init, trainable, and survive save/load (real checkpoints, GPU)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from outliers.losses import OutlierRegularizer, fused_kl, peak_penalty
from outliers.models import iter_residual_norms, load_model
from outliers.retrofit import RetrofitConfig, apply_retrofit, load_retrofit, save_retrofit

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

TEXT = "The quick brown fox jumps over the lazy dog. " * 12

CONFIGS = {
    "qwen3.5-0.8b": [RetrofitConfig(gated_norm=True), RetrofitConfig(linear_bias=True)],
    "qwen3-0.6b": [
        RetrofitConfig(gated_norm=True),
        RetrofitConfig(linear_bias=True),
        RetrofitConfig(attn_gate="headwise"),
        RetrofitConfig(attn_gate="elementwise", gated_norm=True, linear_bias=True),
    ],
}
CASES = [(m, c) for m, cs in CONFIGS.items() for c in cs]


def _ids(tok) -> torch.Tensor:  # noqa: ANN001
    return tok(TEXT, return_tensors="pt").input_ids[:, :96].repeat(2, 1).cuda()


@pytest.mark.parametrize(("name", "cfg"), CASES, ids=[f"{m}-{c}" for m, c in CASES])
def test_identity_at_init_and_roundtrip(name: str, cfg: RetrofitConfig, tmp_path: Path) -> None:
    model, tok = load_model(name)
    ids = _ids(tok)
    with torch.no_grad():
        ref = model(input_ids=ids).logits
    new = apply_retrofit(model, cfg)
    assert new, "retrofit created no parameters"
    with torch.no_grad():
        out = model(input_ids=ids).logits
    assert torch.equal(ref, out), "retrofit is not an exact identity at init"

    # perturb the new parameters, then check save/load reproduces the modified model
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n in new:
            params[n].normal_(0, 0.02)
        pert = model(input_ids=ids).logits
    assert not torch.equal(pert, ref)
    save_retrofit(model, cfg, name, tmp_path / "ckpt.pt")
    loaded, _, cfg2, _ = load_retrofit(tmp_path / "ckpt.pt")
    assert cfg2 == cfg
    with torch.no_grad():
        again = loaded(input_ids=ids).logits
    assert torch.equal(pert, again)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_training_step_with_checkpointing(name: str) -> None:
    """KL + λR backward under non-reentrant gradient checkpointing: new params receive gradients, R is exact."""
    teacher, tok = load_model(name, dtype=torch.float32)
    student, _ = load_model(name, dtype=torch.float32)
    cfg = RetrofitConfig(gated_norm=True, linear_bias=True,
                         attn_gate="headwise" if name.startswith("qwen3-") else None)
    new = apply_retrofit(student, cfg)
    student.train()
    student.requires_grad_(True)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reg = OutlierRegularizer(student, tau=8.0, site="both")
    ids = _ids(tok)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            th = teacher.model(input_ids=ids).last_hidden_state
        with reg.record():
            h = student.model(input_ids=ids, use_cache=False).last_hidden_state
        r, per_norm = reg.collect()
        kl = fused_kl(h, student.lm_head, th, teacher.lm_head, chunk=64)
    assert kl.item() == pytest.approx(0.0, abs=1e-6)  # identical function at init
    assert len(per_norm) == 2 * len(iter_residual_norms(student))

    # R equals a direct computation on the recorded norm inputs and outputs
    captured: list[torch.Tensor] = []
    hooks = [ni.module.register_forward_pre_hook(lambda _m, a: captured.append(a[0].detach()))
             for ni in iter_residual_norms(student)]
    hooks += [ni.module.register_forward_hook(lambda _m, _a, o: captured.append(o.detach()))
              for ni in iter_residual_norms(student)]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        student.model(input_ids=ids, use_cache=False)
    for hk in hooks:
        hk.remove()
    direct = torch.stack([peak_penalty(x, 8.0) for x in captured]).mean()
    assert r.item() == pytest.approx(direct.item(), rel=1e-4)
    assert r.item() > 0

    (kl + 1e-2 * r).backward()
    params = dict(student.named_parameters())
    for kind in ("gn_up", "bias", "ga_proj"):
        names = [n for n in new if kind in n]
        if names:
            assert any(params[n].grad is not None and params[n].grad.abs().sum() > 0 for n in names), kind
    assert params["model.embed_tokens.weight"].grad is not None
    reg.remove()
