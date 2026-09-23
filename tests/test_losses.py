from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from outliers.losses import fused_ce, fused_kl, peak_penalty

pytestmark = pytest.mark.gpu


def _heads(v: int, d: int, bias: bool) -> tuple[nn.Linear, nn.Linear]:
    torch.manual_seed(0)
    s = nn.Linear(d, v, bias=bias).cuda()
    t = nn.Linear(d, v, bias=False).cuda()
    with torch.no_grad():
        s.weight.mul_(3.0)
        t.weight.copy_(s.weight + 0.5 * torch.randn_like(s.weight))
        if bias:
            s.bias.normal_()
    return s, t


def _bf(x: torch.Tensor) -> torch.Tensor:
    """bf16-rounded values, fp32 compute (the fused loss uses bf16 inputs with fp32 accumulation/output)."""
    return x.bfloat16().float()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def _ref_kl(h: torch.Tensor, s: nn.Linear, th: torch.Tensor, t: nn.Linear) -> torch.Tensor:
    ls = F.linear(_bf(h), _bf(s.weight))
    if s.bias is not None:
        ls = ls + s.bias.float()
    lt = F.linear(_bf(th), _bf(t.weight))
    lps, lpt = ls.log_softmax(-1), lt.log_softmax(-1)
    return (lpt.exp() * (lpt - lps)).sum(-1).mean()


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("chunk", [7, 64, 1000])
def test_fused_kl_matches_full(bias: bool, chunk: int) -> None:
    n, d, v = 50, 32, 300
    s, t = _heads(v, d, bias)
    h = torch.randn(n, d, device="cuda", requires_grad=True)
    th = torch.randn(n, d, device="cuda")
    loss = fused_kl(h, s, th, t, chunk=chunk)
    loss.backward()
    g_h, g_w = h.grad.clone(), s.weight.grad.clone()
    g_b = s.bias.grad.clone() if bias else None
    h.grad = None
    s.zero_grad()
    ref = _ref_kl(h, s, th, t)
    ref.backward()
    assert loss.item() == pytest.approx(ref.item(), rel=1e-5, abs=1e-6)
    # gradients go through bf16 (like autocast backward): compare with a norm-relative tolerance
    assert _rel(g_h, h.grad) < 2e-2
    assert _rel(g_w, s.weight.grad) < 2e-2
    if bias:
        assert _rel(g_b, s.bias.grad) < 1e-5


def test_fused_kl_zero_when_identical() -> None:
    s, _ = _heads(300, 32, False)
    h = torch.randn(40, 32, device="cuda", requires_grad=True)
    loss = fused_kl(h, s, h.detach(), s, chunk=16)
    loss.backward()
    assert abs(loss.item()) < 1e-6
    assert h.grad.abs().max().item() < 1e-6


def test_fused_ce_matches_full() -> None:
    n, d, v = 50, 32, 300
    s, _ = _heads(v, d, True)
    h = torch.randn(n, d, device="cuda", requires_grad=True)
    y = torch.randint(0, v, (n,), device="cuda")
    loss = fused_ce(h, s, y, chunk=13)
    loss.backward()
    g_h, g_w = h.grad.clone(), s.weight.grad.clone()
    h.grad = None
    s.zero_grad()
    logits = F.linear(_bf(h), _bf(s.weight)) + s.bias.float()
    ref = F.cross_entropy(logits, y)
    ref.backward()
    assert loss.item() == pytest.approx(ref.item(), rel=1e-5)
    assert _rel(g_h, h.grad) < 2e-2
    assert _rel(g_w, s.weight.grad) < 2e-2


def test_peak_penalty() -> None:
    x = torch.ones(2, 3, 64)
    assert peak_penalty(x, 8.0).item() == 0.0
    x[..., 0] = 100.0  # one dominant dim: u_0 is just below √64 = 8, so only the τ=4 hinge is active
    u0 = 100.0 / ((100.0**2 + 63) / 64) ** 0.5
    expected = max(u0 - 4.0, 0.0) ** 2
    assert peak_penalty(x, 4.0).item() == pytest.approx(expected, rel=1e-5)
