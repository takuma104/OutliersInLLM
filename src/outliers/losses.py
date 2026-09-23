"""Phase 2 losses: fused chunked lm_head KL / CE and the outlier regularizer R (plan §5.2).

The fused losses never materialize [N, V] logits for the whole batch: the forward pass walks over token chunks,
computes the loss and its gradients w.r.t. the hidden states / lm_head weight / bias chunk by chunk, and the
backward pass only rescales the stored gradients (the approach of Liger's fused linear losses).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from outliers.models import iter_residual_norms
from outliers.retrofit import linear_bias


class _FusedLinearLoss(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        h: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        teacher_h: torch.Tensor | None,
        teacher_weight: torch.Tensor | None,
        teacher_bias: torch.Tensor | None,
        targets: torch.Tensor | None,
        chunk: int,
    ) -> torch.Tensor:
        n = h.shape[0]
        cd = torch.bfloat16
        w = weight.to(cd)
        tw = None if teacher_weight is None else teacher_weight.to(cd)
        grad_h = torch.empty(h.shape, device=h.device, dtype=torch.float32)
        grad_w = torch.zeros(weight.shape, device=weight.device, dtype=torch.float32)
        grad_b = None if bias is None else torch.zeros(bias.shape, device=bias.device, dtype=torch.float32)
        total = torch.zeros((), device=h.device, dtype=torch.float32)
        for s in range(0, n, chunk):
            hc = h[s : s + chunk].to(cd)
            logits = torch.mm(hc, w.T, out_dtype=torch.float32)
            if bias is not None:
                logits += bias.float()
            logp = torch.log_softmax(logits, dim=-1)
            del logits
            if teacher_h is not None:
                tl = torch.mm(teacher_h[s : s + chunk].to(cd), tw.T, out_dtype=torch.float32)
                if teacher_bias is not None:
                    tl += teacher_bias.float()
                lpt = torch.log_softmax(tl, dim=-1)
                del tl
                pt = lpt.exp()
                total += (pt * (lpt - logp)).sum()
                g = logp.exp_() - pt  # d KL(p_t || p_s) / d logits
                del pt, lpt
            else:
                y = targets[s : s + chunk]
                total -= logp.gather(1, y[:, None]).sum()
                g = logp.exp_()
                g[torch.arange(g.shape[0], device=g.device), y] -= 1.0
            g /= n
            gb = g.to(cd)
            grad_h[s : s + chunk] = torch.mm(gb, w, out_dtype=torch.float32)
            torch.addmm(grad_w, gb.T, hc, out_dtype=torch.float32, out=grad_w)
            if grad_b is not None:
                grad_b += g.sum(0)
            del g, gb, logp
        ctx.save_for_backward(grad_h, grad_w, *(() if grad_b is None else (grad_b,)))
        ctx.has_bias = grad_b is not None
        ctx.h_dtype = h.dtype
        ctx.w_dtype = weight.dtype
        return total / n

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # type: ignore[override]
        saved = ctx.saved_tensors
        grad_h = (saved[0] * grad_out).to(ctx.h_dtype)
        grad_w = (saved[1] * grad_out).to(ctx.w_dtype)
        grad_b = (saved[2] * grad_out).to(ctx.w_dtype) if ctx.has_bias else None
        return grad_h, grad_w, grad_b, None, None, None, None, None


def fused_kl(
    h: torch.Tensor, lm_head: nn.Linear, teacher_h: torch.Tensor, teacher_lm_head: nn.Linear, chunk: int = 1024
) -> torch.Tensor:
    """mean over tokens of KL(p_teacher ‖ p_student); h / teacher_h: [..., D]."""
    tb = linear_bias(teacher_lm_head)
    return _FusedLinearLoss.apply(
        h.reshape(-1, h.shape[-1]), lm_head.weight, linear_bias(lm_head),
        teacher_h.reshape(-1, teacher_h.shape[-1]).detach(), teacher_lm_head.weight.detach(),
        None if tb is None else tb.detach(), None, chunk,
    )


def fused_ce(h: torch.Tensor, lm_head: nn.Linear, targets: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """mean cross entropy; ``targets`` aligned with ``h`` (shift beforehand)."""
    return _FusedLinearLoss.apply(
        h.reshape(-1, h.shape[-1]), lm_head.weight, linear_bias(lm_head), None, None, None, targets.reshape(-1), chunk
    )


def peak_penalty(x: torch.Tensor, tau: float, eps: float = 1e-6, first_weight: float = 1.0) -> torch.Tensor:
    """Weighted mean over tokens of Σ_j ReLU(|u_j| − τ)², u = x / rms(x) (scale-invariant).

    ``first_weight`` re-weights position 0 of each sequence ([B, T, D] input); 1.0 is the plain token mean. A plain
    mean dilutes a first-token massive activation by 1/T, so C2 (Qwen3) can raise it.
    """
    xf = x.float()
    u = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    p = F.relu(u.abs() - tau).square().sum(-1)
    if first_weight == 1.0:
        return p.mean()
    w = torch.ones_like(p)
    w[..., 0] = first_weight
    return (p * w).sum() / w.sum()


RegSite = Literal["in", "out", "both"]


class OutlierRegularizer:
    """R = mean over residual norms (and sites) of ``peak_penalty`` (plan §5.2).

    ``site``: "in" = the norm input x (the residual stream; the plan's definition), "out" = the norm module's output
    (after GatedNorm's gate, i.e. what the next Linears read and what gets quantized), "both" = both. With "in" only,
    a gate can shrink the residual sink yet re-amplify the same dims after the norm (seen in the C1 pilot).

    The hooks compute the penalty whenever ``active`` and grad is enabled, so that gradient-checkpoint recomputation
    replays exactly the same ops; the value is only collected while ``recording`` (the first forward).
    """

    def __init__(self, model: nn.Module, tau: float = 8.0, first_weight: float = 1.0, site: RegSite = "in") -> None:
        self.tau = tau
        self.first_weight = first_weight
        self.active = False
        self.recording = False
        self._values: list[torch.Tensor] = []
        self._names: list[str] = []
        self._handles = []
        for ni in iter_residual_norms(model):
            if site in ("in", "both"):
                self._handles.append(ni.module.register_forward_pre_hook(self._pre_hook(ni.name + ":in")))
            if site in ("out", "both"):
                self._handles.append(ni.module.register_forward_hook(self._post_hook(ni.name + ":out")))

    def _add(self, name: str, x: torch.Tensor) -> None:
        if not (self.active and torch.is_grad_enabled()):
            return
        pen = peak_penalty(x, self.tau, first_weight=self.first_weight)
        if self.recording:
            self._values.append(pen)
            self._names.append(name)

    def _pre_hook(self, name: str):  # noqa: ANN202
        def hook(_m: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            self._add(name, args[0])

        return hook

    def _post_hook(self, name: str):  # noqa: ANN202
        def hook(_m: nn.Module, _args: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            self._add(name, out)

        return hook

    @contextlib.contextmanager
    def record(self) -> Iterator[None]:
        """Active + recording for the forward pass; call ``collect`` afterwards (before backward)."""
        self._values, self._names = [], []
        self.active, self.recording = True, True
        try:
            yield
        finally:
            self.recording = False

    def collect(self) -> tuple[torch.Tensor, dict[str, float]]:
        if not self._values:
            raise RuntimeError("no penalties recorded (grad disabled or regularizer inactive?)")
        stacked = torch.stack(self._values)
        per_norm = dict(zip(self._names, stacked.detach().float().tolist()))
        return stacked.mean(), per_norm

    def deactivate(self) -> None:
        self.active = False

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
