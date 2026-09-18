"""Build a dataset-mode (native-rate) DipCo-DEV manifest, so a composite DEV
metric (mean over AMI-dev, DipCo-dev, vox_sel) can be computed via the same
SpeakerCountDataset path train.py's own val_loader and scripts/
eval_per_recording.py already use for AMI-dev/vox_sel -- no upsampling, no
RTTM/wav rendering, scored at the model's own frame rate.

Why a new script instead of reusing dipco_eval: dipco_eval
(data_ext/nemo_eval/dipco_eval) is DiPCo's official EVAL split (sessions
S01/S03/S06/S07/S08), already spent as ZipCount's held-out corpus for the
unified 4-corpus protocol and the duration-decoder ablation. Using it again
here as a tuning/DEV signal would contaminate it for any future confirmatory
report. DiPCo has its own separate, non-overlapping official DEV split
(S02/S04/S05/S09/S10, data/manifests/dipco_mdm/*_dev.jsonl.gz) that this
project has never touched -- verified zero session-id overlap with eval.

Reuses windows_for_set/cap_silence from src/data/prepare_meeting.py (the
script that built data/meetings/val_manifest.json, i.e. AMI-dev) verbatim,
so DipCo-dev is windowed/labeled/silence-capped identically to AMI-dev:
30s windows, sups_to_frame_counts ground truth (same protocol as
dipco_eval's RTTM, since DiPCo has no word-level alignment either -- both
fall back to sup_speech_intervals' full-segment-span path), 10% max
silence-window fraction, single-track source_offset recipe (no audio
copied to disk).

Usage:
  PYTHONPATH=. python scripts/prepare_dipco_dev.py \
      --dipco-manifest-dir data/manifests/dipco_mdm --out-dir data/dipco_dev
"""
import argparse
import json
import os
import random

from src.data.prepare_meeting import cap_silence, windows_for_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dipco-manifest-dir", default="data/manifests/dipco_mdm")
    ap.add_argument("--out-dir", default="data/dipco_dev")
    ap.add_argument("--win", type=float, default=30.0)
    ap.add_argument("--min-win", type=float, default=5.0)
    ap.add_argument("--max-silence-frac", type=float, default=0.10)
    ap.add_argument("--limit", type=int, default=0, help="recordings (smoke test)")
    args = ap.parse_args()

    rec_manifest = os.path.join(args.dipco_manifest_dir, "dipco-mdm_recordings_dev.jsonl.gz")
    sup_manifest = os.path.join(args.dipco_manifest_dir, "dipco-mdm_supervisions_dev.jsonl.gz")
    for p in (rec_manifest, sup_manifest):
        if not os.path.exists(p):
            raise SystemExit(f"missing manifest: {p}")

    labels_dir = os.path.join(args.out_dir, "labels_dev")
    os.makedirs(labels_dir, exist_ok=True)

    items, cls = windows_for_set(
        "dipco_dev", rec_manifest, sup_manifest, labels_dir,
        args.win, args.min_win, args.limit,
    )
    if not items:
        raise RuntimeError("No DipCo-dev windows were created; check manifest paths.")
    rec_ids = sorted({it["id"].rsplit("_w", 1)[0] for it in items})
    if set(rec_ids) - {"dipco_dev_S02", "dipco_dev_S04", "dipco_dev_S05", "dipco_dev_S09", "dipco_dev_S10"}:
        raise RuntimeError(
            f"unexpected recording ids {rec_ids} -- expected exactly DiPCo's official "
            "dev split (S02/S04/S05/S09/S10); refusing to write a manifest that might "
            "have picked up an eval-split recording."
        )

    items, n_dropped = cap_silence(items, args.max_silence_frac)
    random.Random(17).shuffle(items)

    mpath = os.path.join(args.out_dir, "dipco_dev_manifest.json")
    with open(mpath, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    tot = sum(cls.values())
    dist = {k: round(cls[k] / tot, 4) for k in sorted(cls)}
    hours = sum(it["duration"] for it in items) / 3600
    print(f"[dipco_dev] {len(items)} windows ({hours:.1f}h, dropped {n_dropped} extra "
          f"silence windows) from recordings {rec_ids} -> {mpath}")
    print(f"[dipco_dev] frame dist (before silence cap): {dist}")


if __name__ == "__main__":
    main()
