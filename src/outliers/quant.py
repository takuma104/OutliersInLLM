"""RTN fake quantization (INT8/INT4/FP8 E4M3/NVFP4), SQNR and model-level fake-quant patching (plan M6, M8, §4.1 F).

All quantizers are symmetric round-to-nearest and group along the last dimension:
activations [N, D_in] -> per-token, weights [D_out, D_in] -> per-output-channel.
They compute in fp32 and return the input dtype.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import torch
from torch import nn

from outliers.models import NON_QUANT_KINDS, iter_linears

FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0
_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# ties go to the even-mantissa neighbour (0, 1, 2, 4): these midpoints round up
_E2M1_TIES_UP = (0.75, 1.75, 3.5)

Quantizer = Callable[[torch.Tensor], torch.Tensor]


def _grouped(x: torch.Tensor, group_size: int | None) -> torch.Tensor:
    if group_size is None:
        return x.reshape(-1, x.shape[-1])
    if x.shape[-1] % group_size:
        raise ValueError(f"last dim {x.shape[-1]} not divisible by group size {group_size}")
    return x.reshape(-1, group_size)


def _safe_scale(s: torch.Tensor) -> torch.Tensor:
    return torch.where(s > 0, s, torch.ones_like(s))


def quant_int(x: torch.Tensor, bits: int, group_size: int | None = None) -> torch.Tensor:
    """Symmetric absmax INT quantization with levels in [-(2^(b-1)-1), 2^(b-1)-1]."""
    qmax = 2 ** (bits - 1) - 1
    g = _grouped(x.float(), group_size)
    scale = _safe_scale(g.abs().amax(-1, keepdim=True) / qmax)
    q = torch.clamp(torch.round(g / scale), -qmax, qmax) * scale
    return q.reshape(x.shape).to(x.dtype)


def quant_fp8_e4m3(x: torch.Tensor, group_size: int | None = None) -> torch.Tensor:
    g = _grouped(x.float(), group_size)
    scale = _safe_scale(g.abs().amax(-1, keepdim=True) / FP8_E4M3_MAX)
    q = (g / scale).to(torch.float8_e4m3fn).float() * scale
    return q.reshape(x.shape).to(x.dtype)


def round_e2m1(y: torch.Tensor) -> torch.Tensor:
    """Round to the nearest FP4 E2M1 value (saturating at ±6, ties to even mantissa)."""
    grid = torch.tensor(_E2M1_GRID, device=y.device, dtype=y.dtype)
    mids = torch.tensor(_E2M1_MIDPOINTS, device=y.device, dtype=y.dtype)
    a = y.abs()
    idx = torch.bucketize(a, mids, right=False)
    ties_up = torch.isin(a, torch.tensor(_E2M1_TIES_UP, device=y.device, dtype=y.dtype))
    idx = torch.where(ties_up, idx + 1, idx)
    return torch.sign(y) * grid[idx]


def quant_nvfp4(x: torch.Tensor, block: int = 16, per_row_global: bool = True) -> torch.Tensor:
    """NVFP4: E2M1 values, FP8 E4M3 scale per ``block`` elements, fp32 global scale.

    ``per_row_global=True`` makes the global (1st-stage) scale dynamic per row, i.e. per token for activations
    (GatedNorm paper App. A.4.2); ``False`` uses one per-tensor global scale (weights).
    """
    xf = x.float().reshape(-1, x.shape[-1])
    if per_row_global:
        amax = xf.abs().amax(-1, keepdim=True)
    else:
        amax = xf.abs().amax().reshape(1, 1)
    g_scale = _safe_scale(amax / (FP8_E4M3_MAX * FP4_E2M1_MAX))  # [R,1] or [1,1]
    b = xf.reshape(xf.shape[0], -1, block)
    b_amax = b.abs().amax(-1, keepdim=True)
    b_scale = (b_amax / FP4_E2M1_MAX) / g_scale.unsqueeze(-1)
    b_scale = b_scale.to(torch.float8_e4m3fn).float() * g_scale.unsqueeze(-1)
    q = round_e2m1(b / _safe_scale(b_scale)) * b_scale
    return q.reshape(x.shape).to(x.dtype)


# --- format registry -------------------------------------------------------------------------------------------------

ACT_FORMATS: dict[str, Quantizer] = {
    "int8_tok": lambda x: quant_int(x, 8),
    "int4_tok": lambda x: quant_int(x, 4),
    "fp8_tok": lambda x: quant_fp8_e4m3(x),
    "nvfp4": lambda x: quant_nvfp4(x, per_row_global=True),
}

WEIGHT_FORMATS: dict[str, Quantizer] = {
    "int8_ch": lambda w: quant_int(w, 8),
    "int4_ch": lambda w: quant_int(w, 4),
    "int4_g128": lambda w: quant_int(w, 4, group_size=128),
    "fp8_ch": lambda w: quant_fp8_e4m3(w),
    "nvfp4": lambda w: quant_nvfp4(w, per_row_global=False),
}


@dataclass(frozen=True)
class QuantScheme:
    name: str
    weight: str | None
    act: str | None


SCHEMES: dict[str, QuantScheme] = {
    s.name: s
    for s in (
        QuantScheme("W8A8", "int8_ch", "int8_tok"),
        QuantScheme("W8A8-FP8", "fp8_ch", "fp8_tok"),
        QuantScheme("W4A16", "int4_g128", None),
        QuantScheme("W4A8", "int4_g128", "int8_tok"),
        QuantScheme("W4A4", "nvfp4", "nvfp4"),
        QuantScheme("A8", None, "int8_tok"),
        QuantScheme("A4", None, "nvfp4"),
    )
}


def sqnr_db(x: torch.Tensor, xq: torch.Tensor) -> float:
    x = x.float()
    noise = (x - xq.float()).square().sum()
    return float(10.0 * torch.log10(x.square().sum() / noise.clamp_min(1e-30)))


def rel_err(y_ref: torch.Tensor, y: torch.Tensor) -> float:
    y_ref = y_ref.float()
    return float((y_ref - y.float()).norm() / y_ref.norm().clamp_min(1e-30))


# --- model-level fake quantization -----------------------------------------------------------------------------------


@contextlib.contextmanager
def fake_quantize(
    model: nn.Module,
    scheme: QuantScheme,
    kinds: Iterable[str] | None = None,
    act_kinds: Iterable[str] | None = None,
) -> Iterator[None]:
    """Temporarily fake-quantize nn.Linear weights (in place) and inputs (pre-hook).

    ``kinds`` restricts both weight and activation quantization to those module kinds; ``act_kinds`` further
    restricts activation quantization (e.g. per-kind bottleneck search). lm_head and GDN in_proj_a/b are never
    quantized (plan §3.3).
    """
    kind_set = None if kinds is None else set(kinds)
    act_set = None if act_kinds is None else set(act_kinds)
    targets = [
        li
        for li in iter_linears(model)
        if li.kind not in NON_QUANT_KINDS and (kind_set is None or li.kind in kind_set)
    ]
    saved: list[tuple[nn.Linear, torch.Tensor]] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        if scheme.weight is not None:
            wq = WEIGHT_FORMATS[scheme.weight]
            for li in targets:
                w = li.module.weight
                saved.append((li.module, w.data.clone()))
                w.data.copy_(wq(w.data))
        if scheme.act is not None:
            aq = ACT_FORMATS[scheme.act]

            def pre_hook(_m: nn.Module, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
                return (aq(args[0]),) + args[1:]

            for li in targets:
                if act_set is None or li.kind in act_set:
                    handles.append(li.module.register_forward_pre_hook(pre_hook))
        yield
    finally:
        for h in handles:
            h.remove()
        for mod, w in saved:
            mod.weight.data.copy_(w)
