"""Select the C4 probe documents shared by all Phase 1 models (plan §3.2) and save their indices.

The index file is tracked in git (data/) so the probe set is reproducible; token tensors are rebuilt on demand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from outliers.data import C4_VALIDATION_SHARD, build_probe_set, c4_validation_texts, select_probe_docs
from outliers.models import MODEL_IDS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/probe_c4_docs.json"))
    ap.add_argument("--n-docs", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    models = ["qwen3.5-0.8b", "qwen3-0.6b"]
    toks = [AutoTokenizer.from_pretrained(MODEL_IDS[m]) for m in models]
    texts = c4_validation_texts()
    idx = select_probe_docs(texts, toks, n_docs=args.n_docs, seq_len=args.seq_len, seed=args.seed)
    stats = {}
    for m, tok in zip(models, toks):
        ps = build_probe_set(tok, texts, idx, args.seq_len)
        stats[m] = {
            "delim_frac": float((ps.categories == 1).float().mean()),
            "chars_per_seq_mean": sum(len(tok.decode(r)) for r in ps.input_ids) / len(idx),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"dataset": "allenai/c4", "data_file": C4_VALIDATION_SHARD, "seed": args.seed, "seq_len": args.seq_len,
             "models": stats, "doc_indices": idx},
            indent=1,
        )
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
