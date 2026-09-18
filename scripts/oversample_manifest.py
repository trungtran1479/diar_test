"""Oversample selected sources in a JSONL manifest.

Stage-2b: the combined manifest is ~91% LibriMix, so the real-meeting
windows (ami_train/nsf_train) contribute ~1.5 of every 16-sample batch and
their gradient is drowned out. Repeating them N times lifts the real-data
share (x4: 12.7k -> 51k of 177k windows, ~29% of each batch).

Usage:
    python scripts/oversample_manifest.py \
        --in  data/combined/train_manifest.json \
        --out data/combined/train_manifest_realx4.json \
        --sources ami_train,nsf_train --times 4

Optionally boost the rarest class further: matched windows whose labels
contain >= --label3-frac of frames with count >= 3 get --label3-times
copies instead (3+ overlap is ~2.5% of real frames; it needs data, not
just class weights).
"""
import argparse
import json
import random

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--sources", required=True,
                    help="comma-separated source prefixes to oversample")
    ap.add_argument("--times", type=int, default=4,
                    help="total copies of each matched line (4 = original + 3 extra)")
    ap.add_argument("--label3-frac", type=float, default=0.0,
                    help="if > 0: matched windows with >= this fraction of 3+ frames "
                         "get --label3-times copies instead of --times")
    ap.add_argument("--label3-times", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    prefixes = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    lines_out = []
    n_matched = n_label3 = n_total = 0
    with open(args.inp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            item = json.loads(line)
            src = item.get("source", "")
            copies = 1
            if src.startswith(prefixes):
                n_matched += 1
                copies = args.times
                if args.label3_frac > 0:
                    lab = np.load(item["label_filepath"])
                    if (lab >= 3).mean() >= args.label3_frac:
                        n_label3 += 1
                        copies = args.label3_times
            lines_out.extend([line] * copies)

    # Shuffle so duplicates are spread out (DataLoader reshuffles anyway,
    # but this keeps head/tail inspection representative).
    random.Random(args.seed).shuffle(lines_out)

    with open(args.out, "w") as f:
        f.write("\n".join(lines_out) + "\n")

    print(f"in: {n_total} lines ({n_matched} matched {prefixes}, "
          f"{n_label3} with >={args.label3_frac:.0%} 3+ frames)")
    print(f"out: {len(lines_out)} lines -> {args.out}")
    n_real_out = (n_matched - n_label3) * args.times + n_label3 * args.label3_times
    print(f"real share: {n_real_out / len(lines_out):.1%}")


if __name__ == "__main__":
    main()
