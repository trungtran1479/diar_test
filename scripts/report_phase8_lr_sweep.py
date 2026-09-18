#!/usr/bin/env python3
"""Tabulate the lr-sweep / extra-seed diagnostic (not an adoption gate).

For each seed, shows the head-only macro_f1 (the a0-winner checkpoint) and
the finetuned macro_f1 + delta at each lr, so the seed-2345 instability
question can be read directly: is any one seed an outlier, and does a lower
lr shrink the finetune deltas' spread.
"""
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
SEEDS = (1234, 2345, 3456, 4567, 5678)
LRS = ("1e-5", "1.5e-5", "3e-5")
LR_TAG = {"1e-5": "lr1e5", "1.5e-5": "lr15e5", "3e-5": "lr3e5"}


def macro(path: Path) -> float:
    with path.open() as f:
        return json.load(f)["pooled"]["macro_f1"]


def finetune_path(seed: int, lr: str) -> Path:
    if lr == "3e-5" and seed in (1234, 2345, 3456):
        return RESULTS / f"p8ft_s{seed}_voxsel.json"
    return RESULTS / f"p8ft{LR_TAG[lr]}_s{seed}_voxsel.json"


def main() -> None:
    head_only = {s: macro(RESULTS / f"p8a0_pyr_s{s}_voxsel.json") for s in SEEDS}

    print(f"{'seed':<6}{'head-only':<11}" + "".join(
        f"lr={lr:<7}Δ{'':<9}" for lr in LRS
    ))
    per_lr_deltas = {lr: [] for lr in LRS}
    for seed in SEEDS:
        row = f"{seed:<6}{head_only[seed]:<11.4f}"
        for lr in LRS:
            path = finetune_path(seed, lr)
            if not path.is_file():
                row += f"{'--':<9}{'--':<10}"
                continue
            value = macro(path)
            delta = value - head_only[seed]
            per_lr_deltas[lr].append(delta)
            row += f"{value:<9.4f}{delta:<+10.4f}"
        print(row)

    print()
    for lr in LRS:
        deltas = per_lr_deltas[lr]
        if not deltas:
            continue
        mean = sum(deltas) / len(deltas)
        signs = {1 if d > 0 else (-1 if d < 0 else 0) for d in deltas}
        same_sign = len(signs) == 1
        delta_range = max(deltas) - min(deltas)
        sample_sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        print(
            f"lr={lr:<7} n={len(deltas)} mean_delta={mean:+.4f} "
            f"delta_sd={sample_sd:.4f} delta_range={delta_range:.4f} "
            f"all_same_sign={same_sign} "
            f"deltas={['%+.4f' % d for d in deltas]}"
        )


if __name__ == "__main__":
    main()
