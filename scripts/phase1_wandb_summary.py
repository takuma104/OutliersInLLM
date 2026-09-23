"""Log the Phase 1 follow-up results (not covered by the per-script wandb runs) as one wandb run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import wandb

WANDB_PROJECT = "OutliersInLLM"
PER_MODEL = ["step_up", "ablate_bias", "rtn_layers_A4-INT"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/phase1"))
    ap.add_argument("--report", type=Path, default=Path("docs/reports/phase1-outliers.md"))
    args = ap.parse_args()

    run = wandb.init(project=WANDB_PROJECT, group="phase1", job_type="summary", name="phase1-summary")
    for m in ("qwen3.5-0.8b", "qwen3-0.6b"):
        for name in PER_MODEL:
            p = args.root / m / f"{name}.parquet"
            if p.exists():
                run.log({f"{m}/{name}": wandb.Table(dataframe=pd.read_parquet(p))})
    dep = args.root / "input_dependence.parquet"
    if dep.exists():
        df = pd.read_parquet(dep)
        run.log({"input_dependence": wandb.Table(dataframe=df.astype({c: str for c in df.columns if df[c].dtype == object}))})
    art = wandb.Artifact("phase1-report", type="report")
    for f in [args.report, args.root / "summary.md"]:
        if f.exists():
            art.add_file(str(f))
    run.log_artifact(art)
    run.finish()


if __name__ == "__main__":
    main()
