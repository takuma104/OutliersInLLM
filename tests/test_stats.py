from __future__ import annotations

import numpy as np
import pytest
import torch

from outliers.data import CAT_DELIM, CAT_FIRST, CAT_OTHER
from outliers.quant import SCHEMES, quant_int, quant_nvfp4
from outliers.stats import (
    AttentionAccumulator,
    LinearInputAccumulator,
    ResidualAccumulator,
    energy_topk,
    rms_normalize,
)


def _lower_median(v: np.ndarray) -> float:
    """torch.median convention: lower of the two middle values."""
    return float(np.sort(v)[(v.size - 1) // 2])


def _planted(b: int = 3, t: int = 64, d: int = 32, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, t, d, generator=g)
    x[:, :, 7] += 20.0  # residual sink dim on all tokens
    x[:, 0, 3] = 500.0  # massive activation on first token
    x[1, 10, 5] = -300.0  # a delimiter spike
    cats = torch.full((b, t), CAT_OTHER)
    cats[:, 0] = CAT_FIRST
    cats[1, 10] = CAT_DELIM
    return x, cats


def test_residual_accumulator_matches_numpy() -> None:
    x, cats = _planted()
    acc = ResidualAccumulator(x.shape[-1], top_k=5)
    # two batches to exercise streaming merge
    acc.update(x[:2], cats[:2], seq_offset=0)
    acc.update(x[2:], cats[2:], seq_offset=2)
    s = acc.summary()
    xn, cn = x.numpy(), cats.numpy()

    assert s["max_first"] == pytest.approx(np.abs(xn[cn == CAT_FIRST]).max())
    assert s["max_delim"] == pytest.approx(300.0)
    assert s["max_other"] == pytest.approx(np.abs(xn[cn == CAT_OTHER]).max())
    assert s["top1"] == pytest.approx(500.0)
    assert (s["top1_pos"], s["top1_dim"]) == (0, 3)
    top3 = np.sort(np.abs(xn).ravel())[::-1][:3]
    assert [s["top1"], s["top2"], s["top3"]] == pytest.approx(top3.tolist())

    rest = xn[cn != CAT_FIRST]
    mean_abs = np.abs(rest).mean(0)
    assert s["sink_dim"] == 7
    assert s["sink_score"] == pytest.approx(mean_abs.max() / _lower_median(mean_abs), rel=1e-5)
    assert s["sink_argmax_frac"] == pytest.approx((np.abs(rest).argmax(-1) == 7).mean())

    u = xn / np.sqrt((xn**2).mean(-1, keepdims=True) + 1e-6)
    peak_rest = np.abs(u).max(-1)[cn != CAT_FIRST]
    assert s["peak_p50"] == pytest.approx(np.quantile(peak_rest, 0.5), rel=1e-5)
    assert s["peak_p99"] == pytest.approx(np.quantile(peak_rest, 0.99), rel=1e-5)
    assert s["peak_first_mean"] == pytest.approx(np.abs(u).max(-1)[:, 0].mean(), rel=1e-5)

    dims = acc.dim_table()
    np.testing.assert_allclose(dims["mean_abs"], mean_abs, rtol=1e-5)
    np.testing.assert_allclose(dims["std"], rest.std(0), rtol=1e-4)
    np.testing.assert_allclose(dims["first_mean"], xn[:, 0].mean(0), rtol=1e-5, atol=1e-6)


def test_rms_normalize_and_energy() -> None:
    x = torch.randn(10, 64)
    x[:, 0] = 100.0
    u = rms_normalize(x)
    assert torch.allclose(u.square().sum(-1), torch.full((10,), 64.0), rtol=1e-4)
    e = energy_topk(u, (1, 4))
    assert torch.all(e[1] > 0.9) and torch.all(e[4] >= e[1])
    assert torch.allclose(e[1], u[:, 0].square() / 64)


def test_linear_accumulator_matches_numpy() -> None:
    torch.manual_seed(0)
    b, t, d, o = 2, 32, 64, 48
    x = torch.randn(b, t, d)
    x[..., 11] *= 30.0  # outlier channel
    x[:, 0] *= 10.0
    cats = torch.full((b, t), CAT_OTHER)
    cats[:, 0] = CAT_FIRST
    w = torch.randn(o, d) * 0.05
    acc = LinearInputAccumulator(d, schemes=[SCHEMES["W8A8"], SCHEMES["W4A4"]])
    acc.update(x, cats, w)
    s = acc.summary()

    xr = x[:, 1:].reshape(-1, d).numpy()
    ch_absmax = np.abs(xr).max(0)
    assert s["absmax_rest"] == pytest.approx(ch_absmax.max())
    assert s["ch_max_over_median"] == pytest.approx(ch_absmax.max() / _lower_median(ch_absmax), rel=1e-5)
    sigma = xr.std()
    assert s["sigma"] == pytest.approx(sigma, rel=1e-4)
    assert s["n_ch_over_6sigma"] == int((ch_absmax > 6 * sigma).sum())
    c = xr - xr.mean(-1, keepdims=True)
    kurt = (c**4).mean(-1) / ((c**2).mean(-1) ** 2) - 3
    assert s["kurt_mean"] == pytest.approx(kurt.mean(), rel=1e-4)
    assert s["kurt_max"] == pytest.approx(kurt.max(), rel=1e-4)

    xa = x.reshape(-1, d)
    q = quant_int(xa, 8)
    sq = 10 * np.log10((xa**2).sum().item() / ((xa - q) ** 2).sum().item())
    assert s["sqnr_global_int8_tok"] == pytest.approx(sq, rel=1e-4)
    nsr_tok = ((xa - q) ** 2).sum(-1) / (xa**2).sum(-1)
    assert s["sqnr_int8_tok"] == pytest.approx(-10 * np.log10(nsr_tok.mean().item()), rel=1e-4)
    y = xa @ w.T
    yq = quant_nvfp4(xa) @ quant_nvfp4(w, per_row_global=False).T
    assert s["out_err_global_W4A4"] == pytest.approx(((y - yq).norm() / y.norm()).item(), rel=1e-4)
    rel_tok = ((y - yq) ** 2).sum(-1) / (y**2).sum(-1)
    assert s["out_err_W4A4"] == pytest.approx(rel_tok.mean().sqrt().item(), rel=1e-4)


def test_attention_accumulator() -> None:
    b, h, hkv, t, hd = 2, 4, 2, 8, 4
    probs = torch.zeros(b, h, t, t)
    probs[:, :2, :, 0] = 1.0  # heads 0,1 fully sink
    probs[:, 2:, torch.arange(t), torch.arange(t)] = 1.0  # heads 2,3 attend to self
    acc = AttentionAccumulator(h, hkv, hd, gated=True)
    acc.update_attn(probs)
    v = torch.ones(b, t, hkv * hd)
    v[:, 0] = 0.0  # first-token value norm 0
    acc.update_v(v)
    acc.update_o_in(torch.ones(b, t, h * hd))
    q = torch.zeros(b, t, h, 2 * hd)
    q[..., hd:] = -10.0  # closed gates
    acc.update_q(q.reshape(b, t, -1))
    df = acc.head_table()
    assert df["is_sink_head"].tolist() == [True, True, False, False]
    assert df["attn_to_first"].tolist() == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert df["attn_to_self"].tolist() == pytest.approx([0.0, 0.0, 1.0, 1.0])
    assert df["v_norm_ratio"].tolist() == pytest.approx([0.0] * h)
    assert df["gate_small_frac_rest"].tolist() == pytest.approx([1.0] * h)
