"""Identity-initialized architecture retrofits (Phase 2): GatedNorm, zero-init Linear biases, attention output gate.

All modifications leave the model's function exactly unchanged at initialization:
- GatedNorm: y' = y ⊙ 2σ(W_up · silu(W_down · y)) with W_up = 0  ->  2σ(0) = 1
- Linear bias: zero-initialized bias on every Linear that reads a residual-norm output
- Attention output gate (GA retrofit): o_proj input ⊙ 2σ(W_g · x̂ + b) with W_g = 0, b = 0
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel

from outliers.models import decoder_layers, iter_linears, layer_types, load_model

GateMode = Literal["headwise", "elementwise"]

# Linear kinds whose input is a residual-norm output (these receive a bias in the bias retrofit)
NORM_READER_KINDS: frozenset[str] = frozenset({"qkv", "in_proj_z", "in_proj_ab", "gate_up", "lm_head"})


class GatedNorm(nn.Module):
    """Wraps an existing RMSNorm and multiplies its output by a low-rank element-wise gate (GatedNorm paper §3.4).

    ``2σ(·)`` equals the paper's σ(·) with the norm weight doubled; with ``gn_up`` zero-initialized the module is
    exactly the identity around the wrapped norm.
    """

    def __init__(self, norm: nn.Module, hidden_size: int, rank: int = 16) -> None:
        super().__init__()
        self.norm = norm
        w = norm.weight
        self.gn_down = nn.Linear(hidden_size, rank, bias=False, device=w.device, dtype=w.dtype)
        self.gn_up = nn.Linear(rank, hidden_size, bias=False, device=w.device, dtype=w.dtype)
        nn.init.zeros_(self.gn_up.weight)

    @property
    def weight(self) -> nn.Parameter:
        return self.norm.weight

    def gate(self, y: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.gn_up(F.silu(self.gn_down(y))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        return y * self.gate(y)


class AttnOutputGate(nn.Module):
    """Gate for the attention output (o_proj input): 2σ(W_g · x̂ + b), zero-initialized (identity)."""

    def __init__(self, hidden_size: int, n_heads: int, head_dim: int, mode: GateMode, like: torch.Tensor) -> None:
        super().__init__()
        self.mode = mode
        self.head_dim = head_dim
        out = n_heads if mode == "headwise" else n_heads * head_dim
        self.ga_proj = nn.Linear(hidden_size, out, bias=True, device=like.device, dtype=like.dtype)
        nn.init.zeros_(self.ga_proj.weight)
        nn.init.zeros_(self.ga_proj.bias)

    def forward(self, x_hat: torch.Tensor) -> torch.Tensor:
        g = 2.0 * torch.sigmoid(self.ga_proj(x_hat))
        if self.mode == "headwise":
            g = g.repeat_interleave(self.head_dim, dim=-1)
        return g


@dataclass(frozen=True)
class RetrofitConfig:
    gated_norm: bool = False
    gated_norm_rank: int = 16
    linear_bias: bool = False
    attn_gate: GateMode | None = None

    @property
    def is_identity(self) -> bool:
        return not (self.gated_norm or self.linear_bias or self.attn_gate)


def _wrap_norms(model: PreTrainedModel, rank: int) -> None:
    d = model.config.hidden_size
    for layer in decoder_layers(model):
        layer.input_layernorm = GatedNorm(layer.input_layernorm, d, rank)
        layer.post_attention_layernorm = GatedNorm(layer.post_attention_layernorm, d, rank)
    model.model.norm = GatedNorm(model.model.norm, d, rank)


def _add_zbias(mod: nn.Module, _args: tuple[Any, ...], out: torch.Tensor) -> torch.Tensor:
    return out + mod.zbias.to(out.dtype)


def _register_zbias(lin: nn.Linear) -> None:
    lin.register_forward_hook(_add_zbias)


def _add_biases(model: PreTrainedModel) -> None:
    """Separate ``zbias`` parameter added after the matmul: F.linear with a bias would switch cuBLAS to a fused
    addmm kernel whose rounding differs, so the model would not stay bit-identical at init."""
    for li in iter_linears(model, include_lm_head=True):
        if li.kind in NORM_READER_KINDS and li.module.bias is None:
            w = li.module.weight
            li.module.zbias = nn.Parameter(torch.zeros(w.shape[0], device=w.device, dtype=w.dtype))
            _register_zbias(li.module)


def linear_bias(lin: nn.Linear) -> torch.Tensor | None:
    """The additive bias of a Linear, whether native or added by the bias retrofit."""
    return lin.bias if lin.bias is not None else getattr(lin, "zbias", None)


def _attach_attn_gates(model: PreTrainedModel, mode: GateMode) -> None:
    cfg = model.config
    if getattr(cfg, "attn_output_gate", False):
        raise ValueError("model already has an attention output gate")
    types = layer_types(model)
    for i, layer in enumerate(decoder_layers(model)):
        if types[i] != "full_attention":
            continue
        attn = layer.self_attn
        attn.out_gate = AttnOutputGate(cfg.hidden_size, cfg.num_attention_heads, attn.head_dim, mode, attn.o_proj.weight)
        _register_gate_hooks(attn)


def _register_gate_hooks(attn: nn.Module) -> None:
    """x̂ (the attention input) is captured before the attention runs and applied at the o_proj input.

    Both hooks fire again, in the same order, when gradient checkpointing recomputes the layer.
    """

    def capture(mod: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        x_hat = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        mod._gate_value = mod.out_gate(x_hat)

    def apply(_m: nn.Module, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        g = attn._gate_value
        return (args[0] * g.to(args[0].dtype),) + args[1:]

    attn.register_forward_pre_hook(capture, with_kwargs=True)
    attn.o_proj.register_forward_pre_hook(apply)


def apply_retrofit(model: PreTrainedModel, cfg: RetrofitConfig) -> list[str]:
    """Modify ``model`` in place; return the names of the newly created parameters.

    Order matters for the bias retrofit: biases are added before GatedNorm so GatedNorm's own Linears stay bias-free.
    """
    before = {n for n, _ in model.named_parameters()}
    if cfg.linear_bias:
        _add_biases(model)
    if cfg.attn_gate is not None:
        _attach_attn_gates(model, cfg.attn_gate)
    if cfg.gated_norm:
        _wrap_norms(model, cfg.gated_norm_rank)
    renamed = {_unwrap_name(n) for n, _ in model.named_parameters()}
    missing = before - renamed
    if missing:
        raise RuntimeError(f"retrofit lost parameters: {sorted(missing)[:5]}")
    return [n for n, _ in model.named_parameters() if _unwrap_name(n) not in before]


def _unwrap_name(name: str) -> str:
    """Parameter name with GatedNorm's wrapping level removed (…input_layernorm.norm.weight -> …input_layernorm.weight)."""
    return name.replace("layernorm.norm.", "layernorm.").replace("model.norm.norm.", "model.norm.")


def save_retrofit(model: PreTrainedModel, cfg: RetrofitConfig, base: str, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"base": base, "config": dataclasses.asdict(cfg), "state_dict": model.state_dict(),
                "extra": extra or {}}, path)


def load_retrofit(
    path: Path, dtype: torch.dtype = torch.bfloat16, device: str = "cuda", attn_impl: str = "sdpa"
) -> tuple[PreTrainedModel, Any, RetrofitConfig, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = RetrofitConfig(**ckpt["config"])
    model, tok = load_model(ckpt["base"], dtype=dtype, attn_impl=attn_impl, device=device)
    apply_retrofit(model, cfg)
    model.load_state_dict({k: v.to(dtype) for k, v in ckpt["state_dict"].items()}, strict=True)
    model.requires_grad_(False)
    return model, tok, cfg, ckpt.get("extra", {}) | {"base": ckpt["base"]}


def gated_norms(model: nn.Module) -> dict[str, GatedNorm]:
    return {n: m for n, m in model.named_modules() if isinstance(m, GatedNorm)}


def attn_gates(model: nn.Module) -> dict[str, AttnOutputGate]:
    return {n: m for n, m in model.named_modules() if isinstance(m, AttnOutputGate)}
