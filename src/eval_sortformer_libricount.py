"""Sortformer on LibriCount (clip-level speaker counting, ground truth 0..10
mapped to {0,1,2,3+}) — same protocol as src/eval_bench.py --bench libricount
for ZipCount. LibriCount has no RTTM/DER ground truth (it's a counting-only
benchmark, not a diarization corpus), so this calls model.diarize() directly
rather than going through the DER-scoring e2e_diarize_speech.py wrapper —
same underlying Sortformer inference, just without RTTM/DER machinery that
doesn't apply here.

Example:
  PYTHONPATH=. python src/eval_sortformer_libricount.py \
      --dir data_ext/libricount/test > logs/sortformer_eval_libricount.log
"""
import argparse
import glob
import os
from collections import Counter

import numpy as np

from src.eval_sortformer_ami import segments_to_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data_ext/libricount/test")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="nvidia/diar_sortformer_4spk-v1")
    args = ap.parse_args()

    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.from_pretrained(args.model, map_location="cuda")
    model.eval()
    print(f"Loaded {args.model}")

    wavs = sorted(glob.glob(os.path.join(args.dir, "*.wav")))
    if args.limit:
        wavs = wavs[:args.limit]
    print(f"LibriCount: {len(wavs)} clips (5 s each), gt 0..10 -> {{0,1,2,3+}}")

    conf = np.zeros((4, 4), dtype=np.int64)
    per_k, per_k_ok = Counter(), Counter()
    mae_sum = 0.0

    for i in range(0, len(wavs), args.batch_size):
        chunk = wavs[i:i + args.batch_size]
        gts = [int(os.path.basename(p).split("_")[0]) for p in chunk]
        segs = model.diarize(audio=chunk, batch_size=len(chunk), verbose=False)
        for gt, sg in zip(gts, segs):
            counts = segments_to_counts(sg, n_frames=500)  # 5 s @ 100 Hz
            clip_pred = int(counts.max())  # max concurrent speakers, capped 3+
            gt_c = min(gt, 3)
            conf[gt_c, clip_pred] += 1
            per_k[gt] += 1
            per_k_ok[gt] += int(clip_pred == gt_c)
            mae_sum += abs(clip_pred - gt_c)

    n = conf.sum()
    acc = np.trace(conf) / n
    print(f"\n===== Sortformer 4spk-v1 — LibriCount (clip-level, decoder=max-concurrent) =====")
    print(f"acc={acc:.3f}  mae(capped)={mae_sum / n:.3f}   n={n}")
    print("confusion (rows=gt capped, cols=pred):")
    for i in range(4):
        row = conf[i] / max(conf[i].sum(), 1)
        print(f"  gt{i}: " + "  ".join(f"{v:.3f}" for v in row))
    print("accuracy by raw speaker count k:")
    print("  " + "  ".join(f"k={k}:{per_k_ok[k]/per_k[k]:.2f}" for k in sorted(per_k)))


if __name__ == "__main__":
    main()
