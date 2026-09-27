"""Per-document mean NLL on the C4-val64 windows, for paired bootstrap comparisons between runs.

Writes <run dir>/eval/c4val64_doc_nll.npy ([64] float64; every window has 1023 predicted tokens, so the mean over
documents equals the pooled mean behind c4val64/ppl).
Usage: python scripts/ptq_docnll.py results/phase3/M0-catq/quantized.pt ... [--fp M0=qwen3-0.6b ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from outliers.data import c4_validation_windows
from outliers.evaluate import token_nll
from outliers.models import load_model
from outliers.retrofit import load_retrofit

FP_SRC = {"M0": "qwen3-0.6b", "M1": "results/phase2/c2/B1-lam3e-3/final.pt", "M3": "results/phase2/c2/B3h-lam3e-3/final.pt"}


@torch.no_grad()
def doc_nll(model: torch.nn.Module, windows: torch.Tensor, batch: int = 8) -> np.ndarray:
    out = [token_nll(model, windows[s : s + batch].cuda()).double().mean(1).cpu()
           for s in range(0, windows.shape[0], batch)]
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="*", type=Path, help="quantized.pt / final.pt files")
    ap.add_argument("--fp", nargs="*", default=[], choices=sorted(FP_SRC), help="also the FP models (results/phase3/fp-<m>)")
    args = ap.parse_args()
    jobs = [(Path(f"results/phase3/fp-{m}/eval"), FP_SRC[m]) for m in args.fp]
    jobs += [(p.parent / "eval", str(p)) for p in args.ckpts]
    windows = None
    for out, src in jobs:
        dest = out / "c4val64_doc_nll.npy"
        if dest.exists():
            continue
        model, tok = (load_retrofit(Path(src))[:2] if src.endswith(".pt") else load_model(src))
        if windows is None:
            windows = c4_validation_windows(tok, 64)
        nll = doc_nll(model, windows)
        out.mkdir(parents=True, exist_ok=True)
        np.save(dest, nll)
        print(f"{src}: c4val64 ppl {np.exp(nll.mean()):.3f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
