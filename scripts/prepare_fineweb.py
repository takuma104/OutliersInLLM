"""Tokenize FineWeb-Edu (sample-10BT) into a packed uint32 token stream per tokenizer (Phase 2 training data).

Documents are concatenated in file order with the tokenizer's EOS appended to each; the trainer cuts the stream
into non-overlapping 2048-token sequences. The training stream comes from shard 000, the held-out stream from 013.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from outliers.models import MODEL_IDS

REPO = "HuggingFaceFW/fineweb-edu"
SHARDS = {"train": "sample/10BT/000_00000.parquet", "heldout": "sample/10BT/013_00000.parquet"}


def write_stream(path: Path, parquet: str, tok, max_tokens: int, batch_docs: int = 2048) -> dict:  # noqa: ANN001
    eos = tok.eos_token_id
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint32, shape=(max_tokens,))
    pos, docs = 0, 0
    pf = pq.ParquetFile(parquet)
    t0 = time.perf_counter()
    for batch in pf.iter_batches(batch_size=batch_docs, columns=["text"]):
        texts = batch.column("text").to_pylist()
        for ids in tok(texts, add_special_tokens=False).input_ids:
            ids.append(eos)
            n = min(len(ids), max_tokens - pos)
            out[pos : pos + n] = ids[:n]
            pos += n
            docs += 1
            if pos >= max_tokens:
                break
        print(f"  {path.name}: {pos / 1e6:.1f}M tokens, {docs} docs, {time.perf_counter() - t0:.0f}s", flush=True)
        if pos >= max_tokens:
            break
    out.flush()
    if pos < max_tokens:
        raise ValueError(f"{parquet} has only {pos} tokens")
    return {"file": parquet, "n_tokens": pos, "n_docs": docs, "eos": eos}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-tokens", type=float, default=320e6)
    ap.add_argument("--heldout-tokens", type=float, default=8e6)
    ap.add_argument("--out", type=Path, default=Path("results/data/fineweb_edu"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_IDS[args.model])
    out = args.out / args.model
    out.mkdir(parents=True, exist_ok=True)
    meta = {"repo": REPO, "tokenizer": MODEL_IDS[args.model]}
    for split, n in (("train", args.train_tokens), ("heldout", args.heldout_tokens)):
        local = hf_hub_download(REPO, SHARDS[split], repo_type="dataset")
        meta[split] = write_stream(out / f"{split}.npy", local, tok, int(n))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
