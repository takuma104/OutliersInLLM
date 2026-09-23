"""Phase 1 §4.1 H (optional): quadratic-form norms ‖U_k‖_F of SwiGLU MLPs and super weights (2603.05498 Fig. 3).

For an MLP y = W_down (silu(W_g h) ⊙ W_u h), treating silu as ≈ linear for spike tokens, output channel k is the
quadratic form hᵀ U_k h with U_k = Σ_j W_down[k, j] w_g,j w_u,jᵀ. Hence
    ‖U_k‖_F² = W_down[k] (G ⊙ U) W_down[k]ᵀ,  G = W_g W_gᵀ, U = W_u W_uᵀ  (both [F, F]).
Spike channels of step-up / step-down blocks show up as k with ‖U_k‖_F ≫ median.

Super weight: in each MLP, the down_proj input channel j* with the largest first-token activation (Phase 1
linear_channels) and the largest |W_down[:, j*]| entry.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from outliers.models import decoder_layers, load_model


@torch.no_grad()
def uk_norms(mlp: torch.nn.Module) -> torch.Tensor:
    wg, wu, wd = (m.weight.double() for m in (mlp.gate_proj, mlp.up_proj, mlp.down_proj))
    m = (wg @ wg.T) * (wu @ wu.T)  # [F, F]
    return ((wd @ m) * wd).sum(-1).clamp_min(0).sqrt().float()  # [D]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", type=Path, default=Path("results/phase1"))
    args = ap.parse_args()

    model, _ = load_model(args.model)
    ch = pd.read_parquet(args.root / args.model / "linear_channels.parquet")
    rows = []
    for i, layer in enumerate(decoder_layers(model)):
        n = uk_norms(layer.mlp)
        top = torch.topk(n, 3)
        down = ch[ch.name == f"model.layers.{i}.mlp.down_proj"].set_index("channel")
        j_first = int(down.absmax_first.idxmax())
        j_rest = int(down.absmax_rest.idxmax())
        wcol = layer.mlp.down_proj.weight[:, j_first].float()
        rows.append(
            {
                "layer": i,
                "uk_top1_dim": int(top.indices[0]),
                "uk_top1_over_median": float(top.values[0] / n.median()),
                "uk_top2_dim": int(top.indices[1]),
                "uk_top2_over_median": float(top.values[1] / n.median()),
                "down_in_first_argmax_ch": j_first,
                "down_in_first_absmax": float(down.absmax_first.max()),
                "down_in_rest_argmax_ch": j_rest,
                "down_in_rest_absmax": float(down.absmax_rest.max()),
                "super_weight_row": int(wcol.abs().argmax()),
                "super_weight_abs": float(wcol.abs().max()),
                "super_weight_over_std": float(wcol.abs().max() / layer.mlp.down_proj.weight.float().std()),
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(args.root / args.model / "step_up.parquet")
    pd.set_option("display.width", 250)
    print(df.to_string())


if __name__ == "__main__":
    main()
