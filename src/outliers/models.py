"""Model loading and structural helpers shared by Qwen3.5 (hybrid GDN) and Qwen3 (full attention)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

MODEL_IDS: dict[str, str] = {
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B-Base",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B-Base",
    "qwen3.5-0.8b-instruct": "Qwen/Qwen3.5-0.8B",
    "qwen3-0.6b-instruct": "Qwen/Qwen3-0.6B",
}

AttnImpl = Literal["sdpa", "eager"]

# nn.Linear leaf name -> normalized kind (plan §4.1 C)
_LINEAR_KIND: dict[str, str] = {
    "q_proj": "qkv",
    "k_proj": "qkv",
    "v_proj": "qkv",
    "in_proj_qkv": "qkv",
    "in_proj_z": "in_proj_z",
    "in_proj_a": "in_proj_ab",
    "in_proj_b": "in_proj_ab",
    "o_proj": "o_proj",
    "out_proj": "o_proj",
    "gate_proj": "gate_up",
    "up_proj": "gate_up",
    "down_proj": "down_proj",
    "lm_head": "lm_head",
    # Phase 2 retrofit modules (outliers.retrofit)
    "gn_down": "gatednorm",
    "gn_up": "gatednorm",
    "ga_proj": "attn_gate",
}

# kinds excluded from quantization (measured only; plan §3.3)
NON_QUANT_KINDS: frozenset[str] = frozenset({"in_proj_ab", "lm_head", "gatednorm", "attn_gate"})


def resolve_model_id(name: str) -> str:
    return MODEL_IDS.get(name, name)


def load_model(
    name: str,
    dtype: torch.dtype = torch.bfloat16,
    attn_impl: AttnImpl = "sdpa",
    device: str | torch.device = "cuda",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal LM (Qwen3.5 resolves to the text-only ``Qwen3_5ForCausalLM``)."""
    model_id = resolve_model_id(name)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation=attn_impl)
    model = model.to(device).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def set_attn_impl(model: PreTrainedModel, attn_impl: AttnImpl) -> None:
    model.set_attn_implementation(attn_impl)


def is_zero_centered(model: PreTrainedModel) -> bool:
    """Qwen3.5 RMSNorm computes x̂·(1+w); Qwen3 computes x̂·w."""
    return model.config.model_type.startswith("qwen3_5")


def decoder_layers(model: PreTrainedModel) -> nn.ModuleList:
    return model.model.layers


def layer_types(model: PreTrainedModel) -> list[str]:
    cfg = model.config
    types = getattr(cfg, "layer_types", None)
    if types is None:
        return ["full_attention"] * cfg.num_hidden_layers
    return list(types)


def linear_kind(module_name: str) -> str | None:
    leaf = module_name.rsplit(".", 1)[-1]
    return _LINEAR_KIND.get(leaf)


def layer_index(module_name: str) -> int | None:
    parts = module_name.split(".")
    for i, p in enumerate(parts[:-1]):
        if p == "layers" and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


@dataclass(frozen=True)
class LinearInfo:
    name: str
    kind: str
    layer: int | None
    module: nn.Linear


def iter_linears(model: PreTrainedModel, include_lm_head: bool = False) -> list[LinearInfo]:
    out: list[LinearInfo] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        kind = linear_kind(name)
        if kind is None:
            raise ValueError(f"unknown linear module: {name}")
        if kind == "lm_head" and not include_lm_head:
            continue
        out.append(LinearInfo(name=name, kind=kind, layer=layer_index(name), module=mod))
    return out


@dataclass(frozen=True)
class NormInfo:
    """A hidden-size RMSNorm that reads the residual stream.

    ``site`` is ``"attn"`` (input_layernorm), ``"mlp"`` (post_attention_layernorm) or ``"final"``.
    ``depth`` orders the residual reads: layer l attn -> 2l, mlp -> 2l+1, final -> 2L.
    """

    name: str
    site: str
    layer: int
    depth: int
    module: nn.Module


def iter_residual_norms(model: PreTrainedModel) -> list[NormInfo]:
    layers = decoder_layers(model)
    out: list[NormInfo] = []
    for i, layer in enumerate(layers):
        out.append(NormInfo(f"model.layers.{i}.input_layernorm", "attn", i, 2 * i, layer.input_layernorm))
        out.append(
            NormInfo(f"model.layers.{i}.post_attention_layernorm", "mlp", i, 2 * i + 1, layer.post_attention_layernorm)
        )
    n = len(layers)
    out.append(NormInfo("model.norm", "final", n, 2 * n, model.model.norm))
    return out


def effective_norm_weight(model: PreTrainedModel, norm: nn.Module) -> torch.Tensor:
    """λ_eff (M4): 1+w for zero-centered Qwen3.5 norms, w otherwise. Returned in fp32."""
    w = norm.weight.detach().float()
    return 1.0 + w if is_zero_centered(model) else w


def token_mixer(layer: nn.Module) -> nn.Module:
    """The attention / GDN submodule of a decoder layer (Qwen3.5 layers own exactly one of the two)."""
    mixer = getattr(layer, "linear_attn", None)
    return mixer if mixer is not None else layer.self_attn
