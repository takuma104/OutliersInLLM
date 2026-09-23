"""Forward-hook wiring from model modules to the streaming accumulators in ``stats``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import PreTrainedModel

from outliers.models import decoder_layers, iter_linears, iter_residual_norms, layer_types, token_mixer
from outliers.quant import QuantScheme
from outliers.stats import AttentionAccumulator, LinearInputAccumulator, ResidualAccumulator


class HookSet:
    """Registers forward / forward-pre hooks and removes them together."""

    def __init__(self) -> None:
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def pre(self, module: nn.Module, fn: Callable[..., Any]) -> None:
        self.handles.append(module.register_forward_pre_hook(fn))

    def post(self, module: nn.Module, fn: Callable[..., Any]) -> None:
        self.handles.append(module.register_forward_hook(fn))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def __enter__(self) -> HookSet:
        return self

    def __exit__(self, *exc: object) -> None:
        self.remove()


def _first_tensor(out: Any) -> torch.Tensor:
    return out[0] if isinstance(out, tuple) else out


@dataclass
class ProbeResults:
    residual: pd.DataFrame
    residual_dims: pd.DataFrame
    linears: pd.DataFrame
    linear_channels: pd.DataFrame
    attention: pd.DataFrame
    tokens: dict[str, np.ndarray] = field(default_factory=dict)


class OutlierProbe:
    """Attach accumulators to a model; call ``set_batch`` before each forward.

    Residual reads are taken at every hidden-size RMSNorm input (input_layernorm = residual before layer l,
    post_attention_layernorm = residual after the token mixer, final norm = last residual). Block outputs are
    the token mixer (attention / GDN) and MLP outputs. Attention stats need ``attn_implementation="eager"``.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        *,
        residual: bool = True,
        blocks: bool = True,
        linears: bool = True,
        quant_error: bool = True,
        attention: bool = False,
        schemes: list[QuantScheme] | None = None,
        keep_tokens: bool = True,
    ) -> None:
        self.model = model
        self.hooks = HookSet()
        self._cats: torch.Tensor | None = None
        self._seq_offset = 0
        cfg = model.config
        d = cfg.hidden_size

        self.residual: dict[str, tuple[dict[str, Any], ResidualAccumulator]] = {}
        if residual:
            for ni in iter_residual_norms(model):
                meta = {"site": f"{ni.site}_norm_in", "name": ni.name, "layer": ni.layer, "depth": ni.depth,
                        "kind": "residual", "sub": ni.site}
                acc = ResidualAccumulator(d, keep_tokens=keep_tokens)
                self.residual[ni.name] = (meta, acc)
                self.hooks.pre(ni.module, self._residual_pre(acc))
        if blocks:
            types = layer_types(model)
            for i, layer in enumerate(decoder_layers(model)):
                mixer_name = "linear_attn" if types[i] == "linear_attention" else "self_attn"
                for sub, mod, depth in (("mixer", token_mixer(layer), 2 * i), ("mlp", layer.mlp, 2 * i + 1)):
                    name = f"model.layers.{i}.{mixer_name if sub == 'mixer' else 'mlp'}"
                    meta = {"site": f"{sub}_out", "name": name, "layer": i, "depth": depth, "kind": "block",
                            "sub": mixer_name if sub == "mixer" else "mlp"}
                    acc = ResidualAccumulator(d, keep_tokens=False)
                    self.residual[name + ".out"] = (meta, acc)
                    self.hooks.post(mod, self._block_post(acc))

        self.linears: dict[str, tuple[dict[str, Any], LinearInputAccumulator]] = {}
        if linears:
            for li in iter_linears(model):
                acc = LinearInputAccumulator(
                    li.module.in_features,
                    act_formats=None if quant_error else {},
                    schemes=schemes if quant_error else [],
                )
                self.linears[li.name] = ({"name": li.name, "kind": li.kind, "layer": li.layer}, acc)
                self.hooks.pre(li.module, self._linear_pre(acc, li.module, quant_error))

        self.attention: dict[int, AttentionAccumulator] = {}
        if attention:
            self._attach_attention()

    # hooks -----------------------------------------------------------------------------------------------------------

    def _residual_pre(self, acc: ResidualAccumulator) -> Callable[..., None]:
        def hook(_m: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            acc.update(args[0], self._cats, self._seq_offset)

        return hook

    def _block_post(self, acc: ResidualAccumulator) -> Callable[..., None]:
        def hook(_m: nn.Module, _args: tuple[Any, ...], out: Any) -> None:
            acc.update(_first_tensor(out), self._cats, self._seq_offset)

        return hook

    def _linear_pre(self, acc: LinearInputAccumulator, mod: nn.Linear, with_w: bool) -> Callable[..., None]:
        def hook(_m: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            acc.update(args[0], self._cats, mod.weight if with_w else None)

        return hook

    def _attach_attention(self) -> None:
        cfg = self.model.config
        types = layer_types(self.model)
        gated = bool(getattr(cfg, "attn_output_gate", False))
        for i, layer in enumerate(decoder_layers(self.model)):
            if types[i] != "full_attention":
                continue
            attn = layer.self_attn
            acc = AttentionAccumulator(cfg.num_attention_heads, cfg.num_key_value_heads, attn.head_dim, gated)
            self.attention[i] = acc

            def attn_hook(_m: nn.Module, _a: tuple[Any, ...], out: Any, acc: AttentionAccumulator = acc) -> None:
                if out[1] is None:
                    raise RuntimeError("attention weights unavailable; load with attn_implementation='eager'")
                acc.update_attn(out[1])

            self.hooks.post(attn, attn_hook)
            self.hooks.post(attn.v_proj, lambda _m, _a, out, acc=acc: acc.update_v(out))
            self.hooks.pre(attn.o_proj, lambda _m, args, acc=acc: acc.update_o_in(args[0]))
            if gated:
                self.hooks.post(attn.q_proj, lambda _m, _a, out, acc=acc: acc.update_q(out))

    # driving ---------------------------------------------------------------------------------------------------------

    def set_batch(self, categories: torch.Tensor, seq_offset: int) -> None:
        self._cats = categories
        self._seq_offset = seq_offset

    def remove(self) -> None:
        self.hooks.remove()

    def __enter__(self) -> OutlierProbe:
        return self

    def __exit__(self, *exc: object) -> None:
        self.remove()

    @torch.no_grad()
    def run(self, input_ids: torch.Tensor, categories: torch.Tensor, batch_size: int) -> None:
        """Forward the base model (no lm_head) over ``input_ids`` [N, T] in batches."""
        dev = next(self.model.parameters()).device
        for s in range(0, input_ids.shape[0], batch_size):
            ids = input_ids[s : s + batch_size].to(dev)
            self.set_batch(categories[s : s + batch_size].to(dev), s)
            self.model.model(input_ids=ids, use_cache=False)

    def results(self) -> ProbeResults:
        res_rows, dim_frames, tokens = [], [], {}
        for key, (meta, acc) in self.residual.items():
            if acc.n_tokens == 0:
                continue
            res_rows.append(meta | acc.summary())
            dim_frames.append(acc.dim_table().assign(name=meta["name"], site=meta["site"], depth=meta["depth"]))
            if acc.keep_tokens:
                for k, v in acc.token_values().items():
                    tokens[f"{key}/{k}"] = v
        lin_rows, ch_frames = [], []
        for meta, acc in self.linears.values():
            if not acc._ready:
                continue
            lin_rows.append(meta | acc.summary())
            ch_frames.append(acc.channel_table().assign(name=meta["name"], kind=meta["kind"], layer=meta["layer"]))
        attn_frames = [acc.head_table().assign(layer=i) for i, acc in self.attention.items() if acc.n_queries]
        return ProbeResults(
            residual=pd.DataFrame(res_rows),
            residual_dims=pd.concat(dim_frames, ignore_index=True) if dim_frames else pd.DataFrame(),
            linears=pd.DataFrame(lin_rows),
            linear_channels=pd.concat(ch_frames, ignore_index=True) if ch_frames else pd.DataFrame(),
            attention=pd.concat(attn_frames, ignore_index=True) if attn_frames else pd.DataFrame(),
            tokens=tokens,
        )
