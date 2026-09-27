"""Phase 3: weight PTQ of an original or retrofitted model (RTN / GPTQ / CAT-Q), saved in the retrofit format.

The output results/phase3/<name>/quantized.pt loads with outliers.retrofit.load_retrofit and is evaluated with
scripts/eval_retrofit.py --ckpt.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path

import torch
from catq.calibration import build_calibration_data
from catq.config import CATQConfig

from outliers.models import load_model
from outliers.ptq import TARGET_SUFFIXES, quantize_catq, quantize_model_sequential, target_linears
from outliers.retrofit import RetrofitConfig, load_retrofit, save_retrofit

CALIB = Path("results/phase3/calib/c4_train_512x2048_seed0.pt")


def load(model_arg: str) -> tuple[torch.nn.Module, object, RetrofitConfig, str]:
    if model_arg.endswith(".pt"):
        model, tok, cfg, extra = load_retrofit(Path(model_arg))
        return model, tok, cfg, extra["base"]
    model, tok = load_model(model_arg)
    return model, tok, RetrofitConfig(), model_arg


def effective_bits(model: torch.nn.Module, weight_bits: float) -> dict[str, float]:
    quantized = sum(m.weight.numel() for layer in model.model.layers for m in target_linears(layer).values())
    total = sum(p.numel() for p in model.parameters())
    tied = model.lm_head.weight is model.model.embed_tokens.weight
    other = total - quantized  # parameters() yields a tied weight once
    return {"quantized_params": quantized, "bf16_params": other, "tied_embeddings": tied,
            "bits_per_quantized_weight": weight_bits,
            "avg_bits_all_params": (quantized * weight_bits + other * 16) / total}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model key (e.g. qwen3-0.6b) or a retrofit checkpoint .pt")
    ap.add_argument("--method", required=True, choices=["rtn-ternary", "gptq-ternary", "rtn-int4", "gptq-int4", "catq"])
    ap.add_argument("--name", required=True, help="output: results/phase3/<name>/quantized.pt")
    ap.add_argument("--calib-samples", type=int, default=None, help="default: 128 for GPTQ, 512 for CAT-Q")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60, help="CAT-Q epochs")
    ap.add_argument("--catq-mode", choices=["ternary", "binary"], default="ternary")
    ap.add_argument("--keep-sigma", type=float, default=None,
                    help="RTN/GPTQ: keep weights with |w| > k·std(W) in bf16 (sparse super-weight control)")
    ap.add_argument("--out-root", type=Path, default=Path("results/phase3"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    out = args.out_root / args.name
    out.mkdir(parents=True, exist_ok=True)
    model, tok, cfg, base = load(args.model)
    n_calib = args.calib_samples or (512 if args.method == "catq" else 128)
    calib = build_calibration_data(tok, num_samples=512, seq_len=2048, seed=0, dataset_name="allenai/c4",
                                   dataset_config="en", cache_path=CALIB)[:n_calib]
    t0 = time.time()
    if args.method == "catq":
        config = CATQConfig(model_name=str(args.model), output_dir=str(out), num_calib_samples=n_calib,
                            epochs=args.epochs, group_size=args.group_size, quant_mode=args.catq_mode,
                            target_suffixes=TARGET_SUFFIXES)
        result = quantize_catq(model, calib, config)
        stats: dict = {"catq_config": dataclasses.asdict(config), "layer_stats": result.layer_stats,
                       "windows": [dataclasses.asdict(w) for w in result.windows]}
        bits = 1.58 if args.catq_mode == "ternary" else 1.0
    else:
        method, grid = args.method.split("-")
        layer_stats = quantize_model_sequential(model, calib, grid, method, args.group_size,
                                                keep_sigma=args.keep_sigma)
        stats = {"layer_stats": layer_stats, "keep_sigma": args.keep_sigma,
                 "n_kept": sum(v for k, v in layer_stats.items() if k.endswith("n_kept"))}
        bits = 1.58 if grid == "ternary" else 4.0
    elapsed = time.time() - t0
    stats |= {"method": args.method, "source": args.model, "calib_samples": n_calib, "elapsed_sec": elapsed,
              "group_size": args.group_size} | effective_bits(model, bits)
    save_retrofit(model, cfg, base, out / "quantized.pt", extra={"run": f"phase3/{args.name}", "quant": args.method})
    (out / "quant_log.json").write_text(json.dumps(stats, indent=2, default=str))
    logging.info("quantized %s with %s in %.1f min -> %s", args.model, args.method, elapsed / 60, out)


if __name__ == "__main__":
    main()
