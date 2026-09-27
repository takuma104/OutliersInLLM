from __future__ import annotations

import pytest
import torch
from catq.config import CATQConfig
from torch import nn

from outliers.models import load_model
from outliers.ptq import RetrofitCATQ, gptq_quantize, rtn_quantize, target_linears, ternary_params
from outliers.retrofit import RetrofitConfig, apply_retrofit


def test_twn_known_values() -> None:
    w = torch.tensor([[1.0, -2.0, 0.1, 0.3]])  # mean|w| = 0.85, Δ = 0.595 -> {1, -2} kept, α = 1.5
    delta, alpha = ternary_params(w)
    assert delta.item() == pytest.approx(0.595)
    assert alpha.item() == pytest.approx(1.5)
    q = rtn_quantize(w, "ternary", group_size=4)
    assert torch.allclose(q, torch.tensor([[1.5, -1.5, 0.0, 0.0]]))


def test_rtn_groups_are_independent() -> None:
    w = torch.cat([torch.randn(3, 128), 100 * torch.randn(3, 128)], dim=1)
    q = rtn_quantize(w, "ternary", group_size=128)
    for g in range(2):
        vals = q[:, g * 128 : (g + 1) * 128]
        for r in range(3):
            assert len(torch.unique(vals[r].abs())) <= 2  # {0, α}


def test_gptq_equals_rtn_with_identity_hessian() -> None:
    torch.manual_seed(0)
    w = torch.randn(16, 256)
    for grid in ("ternary", "int4"):
        q = gptq_quantize(w, torch.eye(256), grid, group_size=128, percdamp=0.0)
        assert torch.allclose(q, rtn_quantize(w, grid, group_size=128), atol=1e-6), grid


@pytest.mark.parametrize("grid", ["ternary", "int4"])
def test_gptq_beats_rtn_on_correlated_inputs(grid: str) -> None:
    torch.manual_seed(0)
    n_in, n_out = 256, 32
    mix = torch.randn(n_in, n_in) / n_in**0.5 + torch.eye(n_in)
    x = torch.randn(4096, n_in) @ mix  # correlated inputs
    w = torch.randn(n_out, n_in)
    h = x.T @ x / x.shape[0]
    q_gptq = gptq_quantize(w, h, grid, group_size=128)
    q_rtn = rtn_quantize(w, grid, group_size=128)
    err = lambda q: ((w - q) @ x.T).norm().item()  # noqa: E731
    assert err(q_gptq) < 0.9 * err(q_rtn)
    if grid == "ternary":
        g = q_gptq.reshape(n_out, -1, 128)
        assert all(len(torch.unique(g[r, k].abs())) <= 2 for r in range(n_out) for k in range(g.shape[1]))


@pytest.mark.gpu
@pytest.mark.slow
def test_catq_wrapping_keeps_attention_gate() -> None:
    model, tok = load_model("qwen3-0.6b")
    new = apply_retrofit(model, RetrofitConfig(attn_gate="headwise", gated_norm=True))
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n in new:
            params[n].normal_(0, 0.05)  # non-identity gates, so dropping them would change the output
    ids = tok("The quick brown fox jumps over the lazy dog. " * 8, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        ref = model(input_ids=ids).logits
    quantizer = RetrofitCATQ(model, CATQConfig(model_name="qwen3-0.6b"))
    for idx in range(len(model.model.layers)):
        quantizer._wrap_layer(idx)
    quantizer._set_fp_mode(list(range(len(model.model.layers))), True)
    with torch.no_grad():
        wrapped = model(input_ids=ids).logits
    assert torch.equal(ref, wrapped), "fp-mode CAT-Q wrappers must reproduce the retrofitted model exactly"

    # baking replaces o_proj once more; the gate must follow it
    layer = model.model.layers[0]
    module = quantizer.wrapped[0]["self_attn.o_proj"]
    linear, _ = module.finalize()
    layer.self_attn.o_proj = linear
    from outliers.retrofit import reattach_attn_gate_hooks

    assert reattach_attn_gate_hooks(model) == 1
    assert layer.self_attn._gate_apply_target is layer.self_attn.o_proj
    assert "self_attn.o_proj" in target_linears(layer)


def test_lora_adapters_identity_then_merge() -> None:
    from outliers.ptq import LoRAAdapters

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.up_proj = nn.Linear(16, 32, bias=False)
            self.mlp.down_proj = nn.Linear(32, 16, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.mlp.down_proj(torch.relu(self.mlp.up_proj(x)))

    torch.manual_seed(0)
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([Layer(), Layer()])
    run = lambda x: model.model.layers[1](model.model.layers[0](x))  # noqa: E731
    x = torch.randn(5, 16)
    ref = run(x)
    lora = LoRAAdapters(model, rank=2)
    assert sum(p.numel() for p in lora.parameters()) == 2 * 2 * (16 + 32 + 32 + 16)
    assert torch.equal(run(x), ref), "B = 0 must leave the model unchanged"
    with torch.no_grad():
        for p in lora.b.values():
            p.normal_()
    adapted = run(x)
    assert not torch.allclose(adapted, ref)
    lora.merge()
    assert torch.allclose(run(x), adapted, atol=1e-5)
