"""Streaming outlier statistics (plan §3.3).

Accumulators consume activations batch by batch and keep only O(D) state (plus optional per-token scalars),
never the activations themselves.

- ResidualAccumulator: residual-stream reads / block outputs  -> M1 (max, top-k), M2 (residual sink), M3 (peak of u)
- LinearInputAccumulator: nn.Linear inputs                     -> M5 (distribution), M6 (fake-quant SQNR / output error)
- AttentionAccumulator: full-attention layers (eager)          -> M7 (sink ratio, first-token value norm, GA gate)
- norm_weight_table / weight_stats_table: static weights       -> M4, M8
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedModel

from outliers.data import CAT_DELIM, CAT_FIRST, CAT_OTHER, CATEGORY_NAMES
from outliers.models import effective_norm_weight, iter_linears, iter_residual_norms
from outliers.quant import ACT_FORMATS, SCHEMES, WEIGHT_FORMATS, QuantScheme, Quantizer, rel_err

N_CATS = len(CATEGORY_NAMES)
ENERGY_TOPK: tuple[int, ...] = (4, 16)
QUANTILES: tuple[float, ...] = (0.5, 0.9, 0.99)


def _quantiles(v: np.ndarray, prefix: str) -> dict[str, float]:
    if v.size == 0:
        return {f"{prefix}_p{int(q * 100)}": float("nan") for q in QUANTILES} | {f"{prefix}_max": float("nan")}
    out = {f"{prefix}_p{int(q * 100)}": float(np.quantile(v, q)) for q in QUANTILES}
    out[f"{prefix}_max"] = float(v.max())
    return out


def rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """u = x / rms(x) along the last dim, in fp32 (‖u‖² = D, |u_j| ≤ √D)."""
    xf = x.float()
    return xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)


def energy_topk(u: torch.Tensor, ks: Sequence[int] = ENERGY_TOPK) -> dict[int, torch.Tensor]:
    """Σ_topk u_j² / D per token for each k."""
    d = u.shape[-1]
    top = u.square().topk(max(ks), dim=-1).values.cumsum(-1)
    return {k: top[..., k - 1] / d for k in ks}


class ResidualAccumulator:
    """M1/M2/M3 for a [B, T, D] activation read from the residual stream (or a block output)."""

    def __init__(self, dim: int, top_k: int = 10, eps: float = 1e-6, keep_tokens: bool = True) -> None:
        self.dim = dim
        self.top_k = top_k
        self.eps = eps
        self.keep_tokens = keep_tokens
        self.n_tokens = 0
        self._state_ready = False
        self._top_vals: torch.Tensor | None = None
        self._top_meta: torch.Tensor | None = None  # [k, 3] (global seq, pos, dim)
        self._medians: list[float] = []
        self._tok: dict[str, list[torch.Tensor]] = {"peak": [], "rms": [], "cat": []} | {
            f"e_top{k}": [] for k in ENERGY_TOPK
        }

    def _init_state(self, device: torch.device) -> None:
        f64 = dict(device=device, dtype=torch.float64)
        self.max_by_cat = torch.zeros(N_CATS, device=device)
        self.count_by_cat = torch.zeros(N_CATS, device=device, dtype=torch.long)
        # per-dim stats over non-first tokens
        self.sum_abs = torch.zeros(self.dim, **f64)
        self.sum = torch.zeros(self.dim, **f64)
        self.sumsq = torch.zeros(self.dim, **f64)
        self.max_abs = torch.zeros(self.dim, device=device)
        self.argmax_hist = torch.zeros(self.dim, device=device, dtype=torch.long)
        # per-dim stats of the first token
        self.sum_first = torch.zeros(self.dim, **f64)
        self.max_abs_first = torch.zeros(self.dim, device=device)
        self.n_rest = 0
        self.n_first = 0
        self._state_ready = True

    @torch.no_grad()
    def update(self, x: torch.Tensor, cats: torch.Tensor, seq_offset: int = 0) -> None:
        b, t, d = x.shape
        if not self._state_ready:
            self._init_state(x.device)
        xf = x.float()
        a = xf.abs()
        cats = cats.to(x.device)

        tok_max = a.amax(-1)  # [B, T]
        for c in range(N_CATS):
            m = cats == c
            if m.any():
                self.max_by_cat[c] = torch.maximum(self.max_by_cat[c], tok_max[m].max())
                self.count_by_cat[c] += int(m.sum())

        # top-k elements over (token, dim)
        flat = a.reshape(-1)
        v, i = flat.topk(min(self.top_k, flat.numel()))
        meta = torch.stack([i // (t * d) + seq_offset, (i // d) % t, i % d], dim=-1)
        if self._top_vals is not None:
            v = torch.cat([self._top_vals, v])
            meta = torch.cat([self._top_meta, meta])
        order = v.topk(min(self.top_k, v.numel())).indices
        self._top_vals, self._top_meta = v[order], meta[order]
        self._medians.append(float(flat.median()))

        rest = cats != CAT_FIRST
        xr = xf[rest]  # [N_rest, D]
        ar = a[rest]
        self.sum_abs += ar.sum(0, dtype=torch.float64)
        self.sum += xr.sum(0, dtype=torch.float64)
        self.sumsq += xr.square().sum(0, dtype=torch.float64)
        self.max_abs = torch.maximum(self.max_abs, ar.amax(0))
        self.argmax_hist += torch.bincount(ar.argmax(-1), minlength=d)
        self.n_rest += xr.shape[0]
        first = cats == CAT_FIRST
        xf1 = xf[first]
        self.sum_first += xf1.sum(0, dtype=torch.float64)
        self.max_abs_first = torch.maximum(self.max_abs_first, xf1.abs().amax(0))
        self.n_first += xf1.shape[0]
        self.n_tokens += b * t

        if self.keep_tokens:
            u = rms_normalize(xf, self.eps)
            self._tok["peak"].append(u.abs().amax(-1).flatten().cpu())
            self._tok["rms"].append(xf.square().mean(-1).sqrt().flatten().cpu())
            self._tok["cat"].append(cats.flatten().to(torch.int8).cpu())
            for k, e in energy_topk(u).items():
                self._tok[f"e_top{k}"].append(e.flatten().cpu())

    def token_values(self) -> dict[str, np.ndarray]:
        return {k: torch.cat(v).numpy() for k, v in self._tok.items() if v}

    def dim_table(self) -> pd.DataFrame:
        n = max(self.n_rest, 1)
        mean = (self.sum / n).float()
        var = (self.sumsq / n - (self.sum / n).square()).clamp_min(0).float()
        return pd.DataFrame(
            {
                "dim": np.arange(self.dim),
                "mean_abs": (self.sum_abs / n).float().cpu().numpy(),
                "mean": mean.cpu().numpy(),
                "std": var.sqrt().cpu().numpy(),
                "max_abs": self.max_abs.cpu().numpy(),
                "argmax_frac": (self.argmax_hist.double() / n).float().cpu().numpy(),
                "first_mean": (self.sum_first / max(self.n_first, 1)).float().cpu().numpy(),
                "first_max_abs": self.max_abs_first.cpu().numpy(),
            }
        )

    def summary(self) -> dict[str, float | int]:
        out: dict[str, float | int] = {"n_tokens": self.n_tokens}
        for c, name in enumerate(CATEGORY_NAMES):
            out[f"max_{name}"] = float(self.max_by_cat[c])
        for r in range(min(3, self.top_k)):
            seq, pos, dim = (int(v) for v in self._top_meta[r])
            out[f"top{r + 1}"] = float(self._top_vals[r])
            out[f"top{r + 1}_seq"], out[f"top{r + 1}_pos"], out[f"top{r + 1}_dim"] = seq, pos, dim
        out["median_abs"] = float(np.mean(self._medians))

        mean_abs = (self.sum_abs / max(self.n_rest, 1)).float()
        med = float(mean_abs.median())
        order = mean_abs.argsort(descending=True)
        out["sink_dim"] = int(order[0])
        out["sink_score"] = float(mean_abs[order[0]]) / max(med, 1e-12)
        out["sink2_dim"] = int(order[1])
        out["sink2_score"] = float(mean_abs[order[1]]) / max(med, 1e-12)
        out["sink_argmax_frac"] = float(self.argmax_hist[order[0]]) / max(self.n_rest, 1)
        top_arg = int(self.argmax_hist.argmax())
        out["argmax_mode_dim"] = top_arg
        out["argmax_mode_frac"] = float(self.argmax_hist[top_arg]) / max(self.n_rest, 1)

        if self.keep_tokens and self._tok["peak"]:
            tv = self.token_values()
            other, delim, first = tv["cat"] == CAT_OTHER, tv["cat"] == CAT_DELIM, tv["cat"] == CAT_FIRST
            rest = ~first
            out |= _quantiles(tv["peak"][rest], "peak")
            out["peak_first_mean"] = float(tv["peak"][first].mean()) if first.any() else float("nan")
            out["peak_delim_p50"] = float(np.median(tv["peak"][delim])) if delim.any() else float("nan")
            out["peak_other_p50"] = float(np.median(tv["peak"][other])) if other.any() else float("nan")
            for k in ENERGY_TOPK:
                out[f"e_top{k}_mean"] = float(tv[f"e_top{k}"][rest].mean())
            out["rms_rest_mean"] = float(tv["rms"][rest].mean())
            out["rms_first_mean"] = float(tv["rms"][first].mean()) if first.any() else float("nan")
        return out


class LinearInputAccumulator:
    """M5 (input distribution) and M6 (activation SQNR, layer-output error) for one nn.Linear."""

    def __init__(
        self,
        dim: int,
        act_formats: Mapping[str, Quantizer] | None = None,
        schemes: Sequence[QuantScheme] | None = None,
        sigma_mult: float = 6.0,
    ) -> None:
        self.dim = dim
        self.act_formats = dict(ACT_FORMATS if act_formats is None else act_formats)
        self.schemes = list(schemes) if schemes is not None else [SCHEMES[n] for n in ("W4A16", "W8A8", "W4A4")]
        self.sigma_mult = sigma_mult
        self._ready = False
        self._wq: dict[str, torch.Tensor] = {}

    def _init_state(self, device: torch.device) -> None:
        f64 = dict(device=device, dtype=torch.float64)
        self.ch_absmax_rest = torch.zeros(self.dim, device=device)
        self.ch_absmax_first = torch.zeros(self.dim, device=device)
        self.ch_sum = torch.zeros(self.dim, **f64)
        self.ch_sumsq = torch.zeros(self.dim, **f64)
        self.n_rest = 0
        self.n_first = 0
        self.kurt_sum = 0.0
        self.kurt_max = float("-inf")
        self.kurt_first_sum = 0.0
        self.tok_absmax_sum = 0.0
        self.sig = torch.zeros((), **f64)
        self.noise = {k: torch.zeros((), **f64) for k in self.act_formats}
        self.y_sq = torch.zeros((), **f64)
        self.y_err = {s.name: torch.zeros((), **f64) for s in self.schemes}
        # token-averaged versions: each token weighs the same (massive tokens do not dominate)
        self.n_all = 0
        self.nsr_tok = {k: torch.zeros((), **f64) for k in self.act_formats}
        self.y_err_tok = {s.name: torch.zeros((), **f64) for s in self.schemes}
        self._ready = True

    @staticmethod
    def token_kurtosis(x: torch.Tensor) -> torch.Tensor:
        """Excess kurtosis across channels for each row of [N, D]."""
        c = x - x.mean(-1, keepdim=True)
        m2 = c.square().mean(-1)
        m4 = c.square().square().mean(-1)
        return m4 / m2.square().clamp_min(1e-30) - 3.0

    @torch.no_grad()
    def update(self, x: torch.Tensor, cats: torch.Tensor, weight: torch.Tensor | None = None) -> None:
        if not self._ready:
            self._init_state(x.device)
        xf = x.float().reshape(-1, x.shape[-1])
        cf = cats.to(x.device).reshape(-1)
        rest, first = cf != CAT_FIRST, cf == CAT_FIRST
        xr, x1 = xf[rest], xf[first]
        self.ch_absmax_rest = torch.maximum(self.ch_absmax_rest, xr.abs().amax(0))
        if x1.numel():
            self.ch_absmax_first = torch.maximum(self.ch_absmax_first, x1.abs().amax(0))
            self.kurt_first_sum += float(self.token_kurtosis(x1).sum())
        self.ch_sum += xr.sum(0, dtype=torch.float64)
        self.ch_sumsq += xr.square().sum(0, dtype=torch.float64)
        k = self.token_kurtosis(xr)
        self.kurt_sum += float(k.sum())
        self.kurt_max = max(self.kurt_max, float(k.max()))
        self.tok_absmax_sum += float(xr.abs().amax(-1).sum())
        self.n_rest += xr.shape[0]
        self.n_first += x1.shape[0]

        # M6: activation fake-quant SQNR (all tokens)
        self.n_all += xf.shape[0]
        sig_tok = xf.square().sum(-1).clamp_min(1e-30)
        self.sig += sig_tok.sum(dtype=torch.float64)
        for name, q in self.act_formats.items():
            err_tok = (xf - q(xf)).square().sum(-1)
            self.noise[name] += err_tok.sum(dtype=torch.float64)
            self.nsr_tok[name] += (err_tok / sig_tok).sum(dtype=torch.float64)
        # M6: layer output relative error
        if weight is not None and self.schemes:
            w = weight.float()
            y = xf @ w.T
            y_tok = y.square().sum(-1).clamp_min(1e-30)
            self.y_sq += y_tok.sum(dtype=torch.float64)
            for s in self.schemes:
                if s.weight is not None and s.weight not in self._wq:
                    self._wq[s.weight] = WEIGHT_FORMATS[s.weight](w)
                wq = self._wq[s.weight] if s.weight is not None else w
                xq = ACT_FORMATS[s.act](xf) if s.act is not None else xf
                e_tok = (y - xq @ wq.T).square().sum(-1)
                self.y_err[s.name] += e_tok.sum(dtype=torch.float64)
                self.y_err_tok[s.name] += (e_tok / y_tok).sum(dtype=torch.float64)

    def channel_table(self) -> pd.DataFrame:
        n = max(self.n_rest, 1)
        mean = self.ch_sum / n
        std = (self.ch_sumsq / n - mean.square()).clamp_min(0).sqrt()
        return pd.DataFrame(
            {
                "channel": np.arange(self.dim),
                "absmax_rest": self.ch_absmax_rest.cpu().numpy(),
                "absmax_first": self.ch_absmax_first.cpu().numpy(),
                "mean": mean.float().cpu().numpy(),
                "std": std.float().cpu().numpy(),
            }
        )

    def summary(self) -> dict[str, float | int]:
        n = max(self.n_rest, 1)
        mean_all = float(self.ch_sum.sum()) / (n * self.dim)
        sigma = (float(self.ch_sumsq.sum()) / (n * self.dim) - mean_all**2) ** 0.5
        absmax_rest = float(self.ch_absmax_rest.max())
        out: dict[str, float | int] = {
            "n_tokens": self.n_rest + self.n_first,
            "absmax": max(absmax_rest, float(self.ch_absmax_first.max())),
            "absmax_rest": absmax_rest,
            "absmax_first": float(self.ch_absmax_first.max()),
            "sigma": sigma,
            "absmax_over_sigma": absmax_rest / max(sigma, 1e-12),
            "ch_max_over_median": absmax_rest / max(float(self.ch_absmax_rest.median()), 1e-12),
            "n_ch_over_6sigma": int((self.ch_absmax_rest > self.sigma_mult * sigma).sum()),
            "kurt_mean": self.kurt_sum / n,
            "kurt_max": self.kurt_max,
            "kurt_first_mean": self.kurt_first_sum / max(self.n_first, 1),
            "tok_absmax_mean": self.tok_absmax_sum / n,
        }
        # sqnr_* / out_err_*: token-averaged (primary); *_global: energy-weighted over all tokens
        na = max(self.n_all, 1)
        for name, noise in self.noise.items():
            out[f"sqnr_{name}"] = float(-10 * torch.log10((self.nsr_tok[name] / na).clamp_min(1e-30)))
            out[f"sqnr_global_{name}"] = float(10 * torch.log10(self.sig / noise.clamp_min(1e-30)))
        for s in self.schemes:
            if float(self.y_sq) > 0:
                out[f"out_err_{s.name}"] = float((self.y_err_tok[s.name] / na).sqrt())
                out[f"out_err_global_{s.name}"] = float((self.y_err[s.name] / self.y_sq).sqrt())
        return out


class AttentionAccumulator:
    """M7 for one full-attention layer.

    Sink ratio follows Gu et al. (2025): per head, the attention to the first token averaged over queries
    (t ≥ 1); a head is a sink head when that exceeds ``threshold``.
    """

    def __init__(self, n_heads: int, n_kv_heads: int, head_dim: int, gated: bool) -> None:
        self.n_heads, self.n_kv_heads, self.head_dim, self.gated = n_heads, n_kv_heads, head_dim, gated
        self.attn_first_sum = torch.zeros(n_heads, dtype=torch.float64)
        self.attn_self_sum = torch.zeros(n_heads, dtype=torch.float64)
        self.n_queries = 0
        self.v_norm_first = torch.zeros(n_kv_heads, dtype=torch.float64)
        self.v_norm_rest = torch.zeros(n_kv_heads, dtype=torch.float64)
        self.o_norm_first = torch.zeros(n_heads, dtype=torch.float64)
        self.o_norm_rest = torch.zeros(n_heads, dtype=torch.float64)
        self.gate_first = torch.zeros(n_heads, dtype=torch.float64)
        self.gate_rest = torch.zeros(n_heads, dtype=torch.float64)
        self.gate_small_rest = torch.zeros(n_heads, dtype=torch.float64)
        self.n_seq = 0
        self.n_rest = 0

    @torch.no_grad()
    def update_attn(self, probs: torch.Tensor) -> None:
        """probs: [B, H, T, T] softmax attention weights."""
        p = probs.float()
        self.attn_first_sum += p[:, :, 1:, 0].sum((0, 2)).double().cpu()
        self.attn_self_sum += p.diagonal(dim1=-2, dim2=-1)[:, :, 1:].sum((0, 2)).double().cpu()
        self.n_queries += p.shape[0] * (p.shape[2] - 1)

    @torch.no_grad()
    def update_v(self, v: torch.Tensor) -> None:
        """v: v_proj output [B, T, Hkv*hd]."""
        n = v.float().reshape(*v.shape[:2], self.n_kv_heads, self.head_dim).norm(dim=-1)  # [B,T,Hkv]
        self.v_norm_first += n[:, 0].sum(0).double().cpu()
        self.v_norm_rest += n[:, 1:].sum((0, 1)).double().cpu()
        self.n_seq += v.shape[0]
        self.n_rest += v.shape[0] * (v.shape[1] - 1)

    @torch.no_grad()
    def update_o_in(self, o: torch.Tensor) -> None:
        """o: o_proj input [B, T, H*hd] (after the output gate for gated attention)."""
        n = o.float().reshape(*o.shape[:2], self.n_heads, self.head_dim).norm(dim=-1)
        self.o_norm_first += n[:, 0].sum(0).double().cpu()
        self.o_norm_rest += n[:, 1:].sum((0, 1)).double().cpu()

    @torch.no_grad()
    def update_q(self, q: torch.Tensor) -> None:
        """q: q_proj output of a gated attention layer, [B, T, H*hd*2] = per-head (query, gate)."""
        g = torch.sigmoid(q.float().reshape(*q.shape[:2], self.n_heads, 2 * self.head_dim)[..., self.head_dim :])
        self.gate_first += g[:, 0].mean(-1).sum(0).double().cpu()
        self.gate_rest += g[:, 1:].mean(-1).sum((0, 1)).double().cpu()
        self.gate_small_rest += (g[:, 1:] < 0.1).float().mean(-1).sum((0, 1)).double().cpu()

    def head_table(self, threshold: float = 0.3) -> pd.DataFrame:
        nq, ns, nr = max(self.n_queries, 1), max(self.n_seq, 1), max(self.n_rest, 1)
        attn_first = (self.attn_first_sum / nq).numpy()
        kv_of = np.arange(self.n_heads) // (self.n_heads // self.n_kv_heads)
        v_first, v_rest = (self.v_norm_first / ns).numpy(), (self.v_norm_rest / nr).numpy()
        df = pd.DataFrame(
            {
                "head": np.arange(self.n_heads),
                "attn_to_first": attn_first,
                "attn_to_self": (self.attn_self_sum / nq).numpy(),
                "is_sink_head": attn_first > threshold,
                "v_norm_first": v_first[kv_of],
                "v_norm_rest": v_rest[kv_of],
                "v_norm_ratio": (v_first / np.maximum(v_rest, 1e-12))[kv_of],
                "o_norm_first": (self.o_norm_first / ns).numpy(),
                "o_norm_rest": (self.o_norm_rest / nr).numpy(),
            }
        )
        if self.gated:
            df["gate_first"] = (self.gate_first / ns).numpy()
            df["gate_rest"] = (self.gate_rest / nr).numpy()
            df["gate_small_frac_rest"] = (self.gate_small_rest / nr).numpy()
        return df


# --- static weight statistics ----------------------------------------------------------------------------------------


def norm_weight_table(model: PreTrainedModel) -> pd.DataFrame:
    """M4: effective RMSNorm scale λ_eff per residual norm and dim (long format)."""
    rows = []
    for ni in iter_residual_norms(model):
        lam = effective_norm_weight(model, ni.module).cpu().numpy()
        rows.append(
            pd.DataFrame(
                {"norm": ni.name, "site": ni.site, "layer": ni.layer, "depth": ni.depth,
                 "dim": np.arange(lam.size), "lambda_eff": lam}
            )
        )
    return pd.concat(rows, ignore_index=True)


def _excess_kurtosis(w: torch.Tensor) -> float:
    c = w - w.mean()
    return float(c.pow(4).mean() / c.square().mean().square() - 3.0)


WEIGHT_ERR_FORMATS: tuple[str, ...] = ("int8_ch", "int4_ch", "int4_g128", "nvfp4")


@torch.no_grad()
def weight_stats_table(model: PreTrainedModel) -> pd.DataFrame:
    """M8: per nn.Linear weight outlier statistics and W-only fake-quant relative error."""
    rows = []
    for li in iter_linears(model, include_lm_head=True):
        w = li.module.weight.float()
        col = w.norm(dim=0)  # per input channel
        row: dict[str, float | int | str | None] = {
            "name": li.name,
            "kind": li.kind,
            "layer": li.layer,
            "out_features": w.shape[0],
            "in_features": w.shape[1],
            "kurtosis": _excess_kurtosis(w),
            "absmax": float(w.abs().max()),
            "absmax_over_std": float(w.abs().max() / w.std()),
            "in_ch_norm_max_over_median": float(col.max() / col.median()),
            "in_ch_norm_argmax": int(col.argmax()),
        }
        for f in WEIGHT_ERR_FORMATS:
            if w.shape[1] % 128 == 0 or f != "int4_g128":
                row[f"rel_err_{f}"] = rel_err(w, WEIGHT_FORMATS[f](w))
        rows.append(row)
    return pd.DataFrame(rows)


def lambda_at(norms: pd.DataFrame, norm_name: str, dims: Sequence[int]) -> list[float]:
    sub = norms[norms["norm"] == norm_name].set_index("dim")["lambda_eff"]
    return [float(sub.loc[d]) for d in dims]

