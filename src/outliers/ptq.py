"""Phase 3 PTQ: ternary / low-bit weight quantizers (RTN, GPTQ) and the CAT-Q adapter for retrofitted models.

All quantizers act on the decoder-layer Linears q/k/v/o/gate/up/down (CAT-Q's target set); embeddings, lm_head,
norms and retrofit gates stay in bf16. Weights are fake-quantized in place (bf16 values on the quantization grid).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from catq.config import CATQConfig
from catq.slider import RunResult, SlidingWindowQuantizer
from torch import nn
from transformers import PreTrainedModel

from outliers.models import decoder_layers
from outliers.retrofit import reattach_attn_gate_hooks

logger = logging.getLogger(__name__)

TARGET_SUFFIXES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
Grid = Literal["ternary", "int4"]


def target_linears(layer: nn.Module) -> dict[str, nn.Linear]:
    return {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear) and n.rsplit(".", 1)[-1] in TARGET_SUFFIXES}


def _check_supported(model: nn.Module) -> None:
    if any(n.endswith("zbias") for n, _ in model.named_parameters()):
        raise NotImplementedError("the zero-bias retrofit is not supported by the PTQ path (hooks live on the Linears)")


# --- grids ----------------------------------------------------------------------------------------------------------


def ternary_params(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """TWN (Li et al., 2016) per row of ``w`` [R, G]: threshold Δ = 0.7·mean|w|, scale α = mean of |w| above Δ."""
    a = w.abs()
    delta = 0.7 * a.mean(-1, keepdim=True)
    mask = a > delta
    alpha = (a * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)
    return delta, alpha


def ternary_round(w: torch.Tensor, delta: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return alpha * torch.sign(w) * (w.abs() > delta)


def int_params(w: torch.Tensor, bits: int = 4) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax(-1, keepdim=True) / qmax
    return torch.where(scale > 0, scale, torch.ones_like(scale))


def int_round(w: torch.Tensor, scale: torch.Tensor, bits: int = 4) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    return torch.clamp(torch.round(w / scale), -qmax, qmax) * scale


def rtn_quantize(w: torch.Tensor, grid: Grid, group_size: int = 128) -> torch.Tensor:
    """Round-to-nearest on ``grid`` with per-(row, group) parameters; returns the input dtype."""
    wf = w.float()
    g = wf.reshape(wf.shape[0], -1, group_size)
    if grid == "ternary":
        delta, alpha = ternary_params(g)
        q = ternary_round(g, delta, alpha)
    else:
        q = int_round(g, int_params(g))
    return q.reshape(w.shape).to(w.dtype)


# --- GPTQ -----------------------------------------------------------------------------------------------------------


@torch.no_grad()
def gptq_quantize(w: torch.Tensor, hessian: torch.Tensor, grid: Grid, group_size: int = 128,
                  percdamp: float = 0.01) -> torch.Tensor:
    """GPTQ (Frantar et al., 2023) with the block size equal to the group size.

    Grid parameters of each group are fitted on the (error-updated) weights when the group is reached.
    ``hessian`` is E[x xᵀ] over calibration inputs of this Linear ([in, in]).
    """
    W = w.float().clone()
    H = hessian.float().clone()
    n_in = W.shape[1]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += percdamp * torch.mean(torch.diag(H)) * torch.eye(n_in, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    for i1 in range(0, n_in, group_size):
        i2 = min(i1 + group_size, n_in)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        if grid == "ternary":
            delta, alpha = ternary_params(W1)
            rnd: Callable[[torch.Tensor], torch.Tensor] = lambda x: ternary_round(x, delta[:, 0], alpha[:, 0])
        else:
            scale = int_params(W1)
            rnd = lambda x: int_round(x, scale[:, 0])
        for i in range(i2 - i1):
            col = W1[:, i]
            q = rnd(col)
            Q1[:, i] = q
            err = (col - q) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q.to(w.dtype)


@torch.no_grad()
def _layer_inputs(model: PreTrainedModel, input_ids: torch.Tensor, batch: int) -> tuple[torch.Tensor, tuple]:
    embed = model.model.embed_tokens
    dev = embed.weight.device
    hidden = torch.cat([embed(input_ids[i : i + batch].to(dev)) for i in range(0, input_ids.shape[0], batch)])
    pos = torch.arange(input_ids.shape[1], device=dev).unsqueeze(0)
    return hidden, model.model.rotary_emb(hidden[:1], pos)


@torch.no_grad()
def quantize_model_sequential(model: PreTrainedModel, input_ids: torch.Tensor, grid: Grid, method: str,
                              group_size: int = 128, batch: int = 4) -> dict[str, float]:
    """Layer-by-layer weight quantization with the quantized-model stream as input (RTN ignores the data).

    For GPTQ, each Linear's Hessian is collected from the current layer's inputs (previous layers already
    quantized); all Linears of a layer use one forward pass with that layer still in full precision.
    """
    _check_supported(model)
    hidden, pos = _layer_inputs(model, input_ids, batch)
    stats: dict[str, float] = {}
    for li, layer in enumerate(decoder_layers(model)):
        linears = target_linears(layer)
        if method == "gptq":
            hess = {n: torch.zeros(m.in_features, m.in_features, device=hidden.device) for n, m in linears.items()}
            count = {n: 0 for n in linears}

            def make_hook(name: str) -> Callable[..., None]:
                def hook(_m: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
                    x = args[0].reshape(-1, args[0].shape[-1]).float()
                    hess[name].addmm_(x.T, x)
                    count[name] += x.shape[0]

                return hook

            handles = [m.register_forward_pre_hook(make_hook(n)) for n, m in linears.items()]
            for i in range(0, hidden.shape[0], batch):
                layer(hidden[i : i + batch], attention_mask=None, position_embeddings=pos)
            for h in handles:
                h.remove()
        for name, lin in linears.items():
            w = lin.weight.data
            if method == "gptq":
                q = gptq_quantize(w, hess[name] / count[name], grid, group_size)
            else:
                q = rtn_quantize(w, grid, group_size)
            stats[f"layer{li}.{name}.rel_err"] = float((q.float() - w.float()).norm() / w.float().norm())
            if grid == "ternary":
                stats[f"layer{li}.{name}.zero_fraction"] = float((q == 0).float().mean())
            lin.weight.data.copy_(q)
        if method == "gptq":
            del hess
        for i in range(0, hidden.shape[0], batch):
            hidden[i : i + batch] = layer(hidden[i : i + batch], attention_mask=None, position_embeddings=pos)
    return stats


# --- CAT-Q ----------------------------------------------------------------------------------------------------------


class RetrofitCATQ(SlidingWindowQuantizer):
    """CAT-Q's sliding-window quantizer, keeping retrofit attention gates attached.

    CAT-Q swaps o_proj for a CATQLinear when a layer enters its first window and for a fresh nn.Linear when baking;
    both swaps would silently drop the attention gate (its hook lives on the o_proj object).
    """

    def _wrap_layer(self, idx: int) -> None:
        super()._wrap_layer(idx)
        reattach_attn_gate_hooks(self.layers[idx])

    def run(self, input_ids: torch.Tensor) -> RunResult:
        result = super().run(input_ids)
        reattach_attn_gate_hooks(self.model)
        return result


def quantize_catq(model: PreTrainedModel, input_ids: torch.Tensor, config: CATQConfig) -> RunResult:
    _check_supported(model)
    return RetrofitCATQ(model, config).run(input_ids)


# --- recovery (P3) --------------------------------------------------------------------------------------------------


class LoRAAdapters(nn.Module):
    """Unmerged low-rank adapters on the quantized Linears: y = W_q x + B A x (B zero-initialized).

    The P3 control for gate-only recovery at a matched trainable-parameter count. Applied through forward hooks, so
    the model's state_dict (and the frozen ternary weights) are untouched until :meth:`merge`, which folds B A into
    the weights (the merged weights are no longer on the quantization grid).
    """

    def __init__(self, model: PreTrainedModel, rank: int) -> None:
        super().__init__()
        self.rank = rank
        self.a = nn.ParameterDict()
        self.b = nn.ParameterDict()
        self._targets: dict[str, nn.Linear] = {}
        self._handles = []
        for li, layer in enumerate(decoder_layers(model)):
            for name, lin in target_linears(layer).items():
                key = f"{li}.{name}".replace(".", "__")
                w = lin.weight
                a = torch.empty(rank, lin.in_features, device=w.device, dtype=torch.float32)
                nn.init.kaiming_uniform_(a, a=5**0.5)
                self.a[key] = nn.Parameter(a)
                self.b[key] = nn.Parameter(torch.zeros(lin.out_features, rank, device=w.device, dtype=torch.float32))
                self._targets[key] = lin
                self._handles.append(lin.register_forward_hook(self._hook(key)))

    def _hook(self, key: str) -> Callable[..., torch.Tensor]:
        def hook(_m: nn.Module, args: tuple[torch.Tensor, ...], out: torch.Tensor) -> torch.Tensor:
            x = args[0]
            return out + F.linear(F.linear(x, self.a[key].to(x.dtype)), self.b[key].to(x.dtype))

        return hook

    @torch.no_grad()
    def merge(self) -> None:
        for key, lin in self._targets.items():
            delta = self.b[key].float() @ self.a[key].float()
            lin.weight.copy_((lin.weight.float() + delta).to(lin.weight.dtype))
        for h in self._handles:
            h.remove()
        self._handles.clear()
