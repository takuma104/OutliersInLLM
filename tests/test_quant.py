from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from outliers.quant import (
    SCHEMES,
    fake_quantize,
    quant_fp8_e4m3,
    quant_int,
    quant_nvfp4,
    round_e2m1,
    sqnr_db,
)


def test_int8_known_values() -> None:
    x = torch.tensor([[127.0, -63.4, 0.49, 10.0]])
    q = quant_int(x, 8)  # scale = 1
    assert torch.equal(q, torch.tensor([[127.0, -63.0, 0.0, 10.0]]))


def test_int4_per_row_scale() -> None:
    x = torch.tensor([[7.0, 3.4, -1.0], [0.7, 0.35, -0.7]])
    q = quant_int(x, 4)
    assert torch.allclose(q, torch.tensor([[7.0, 3.0, -1.0], [0.7, 0.4, -0.7]]))


def test_int_group_size() -> None:
    x = torch.cat([torch.full((1, 128), 1.0), torch.full((1, 128), 100.0)], dim=1)
    x[0, 5] = 0.3
    q = quant_int(x, 4, group_size=128)
    # group 0 scale = 1/7: 0.3 -> round(2.1) = 2 -> 2/7; group 1 is exact
    assert math.isclose(float(q[0, 5]), 2 / 7, rel_tol=1e-6)
    assert torch.allclose(q[0, 128:], x[0, 128:])


def test_int_zero_rows() -> None:
    x = torch.zeros(3, 16)
    assert torch.equal(quant_int(x, 4), x)
    assert torch.equal(quant_nvfp4(x), x)
    assert torch.equal(quant_fp8_e4m3(x), x)


def test_e2m1_rounding() -> None:
    y = torch.tensor([0.0, 0.2, 0.25, 0.3, 0.75, 1.25, 1.6, 1.75, 2.5, 2.6, 3.5, 5.0, 5.1, 7.0, -0.75, -9.0])
    exp = torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 3.0, 4.0, 4.0, 6.0, 6.0, -1.0, -6.0])
    assert torch.equal(round_e2m1(y), exp)


def test_nvfp4_block_exact_grid() -> None:
    # block absmax 6 with global scale making the block scale exactly 1 -> values on the E2M1 grid are exact
    vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 2)
    x = torch.stack([vals, -vals])
    q = quant_nvfp4(x, per_row_global=True)
    assert torch.allclose(q, x)


def test_nvfp4_block_independence() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 64)
    x[:, 0] = 1000.0  # outlier in the first block only
    q = quant_nvfp4(x, per_row_global=False)
    rel_rest = (q[:, 16:] - x[:, 16:]).norm() / x[:, 16:].norm()
    assert rel_rest < 0.15  # other blocks keep their own scale


def test_fp8_matches_torch_cast() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 32)
    q = quant_fp8_e4m3(x)
    s = x.abs().amax(-1, keepdim=True) / 448.0
    ref = (x / s).to(torch.float8_e4m3fn).float() * s
    assert torch.allclose(q, ref)


@pytest.mark.parametrize(
    ("fn", "lo", "hi"),
    [
        (lambda x: quant_int(x, 8), 40.0, 45.0),  # 10log10(12*127^2/3.2^2) ≈ 42.8
        (lambda x: quant_int(x, 4, group_size=128), 17.0, 21.0),
        (lambda x: quant_fp8_e4m3(x), 29.0, 33.0),
        (lambda x: quant_nvfp4(x), 18.0, 22.0),
    ],
)
def test_sqnr_gaussian_ranges(fn, lo: float, hi: float) -> None:
    torch.manual_seed(0)
    x = torch.randn(256, 1024)
    s = sqnr_db(x, fn(x))
    assert lo < s < hi, s


def test_dtype_preserved() -> None:
    x = torch.randn(4, 32, dtype=torch.bfloat16)
    for q in (quant_int(x, 8), quant_fp8_e4m3(x), quant_nvfp4(x)):
        assert q.dtype == torch.bfloat16


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(32, 64, bias=False)
        self.mlp.down_proj = nn.Linear(64, 32, bias=False)
        self.lm_head = nn.Linear(32, 10, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.mlp.down_proj(torch.relu(self.mlp.gate_proj(x))))


def test_fake_quantize_restores_and_scopes() -> None:
    torch.manual_seed(0)
    m = _Tiny()
    x = torch.randn(5, 32)
    w_before = {n: p.clone() for n, p in m.named_parameters()}
    y0 = m(x)
    with fake_quantize(m, SCHEMES["W4A4"]):
        y1 = m(x)
        assert torch.equal(m.lm_head.weight, w_before["lm_head.weight"])
        assert not torch.equal(m.mlp.gate_proj.weight, w_before["mlp.gate_proj.weight"])
    with fake_quantize(m, SCHEMES["A4"], act_kinds=["down_proj"]):
        assert torch.equal(m.mlp.gate_proj.weight, w_before["mlp.gate_proj.weight"])
        y2 = m(x)
    for n, p in m.named_parameters():
        assert torch.equal(p, w_before[n])
    assert torch.equal(m(x), y0)
    assert not torch.allclose(y1, y0)
    h = torch.relu(m.mlp.gate_proj(x))
    ref = m.lm_head(m.mlp.down_proj(quant_nvfp4(h)))
    assert torch.allclose(y2, ref, atol=1e-6)
