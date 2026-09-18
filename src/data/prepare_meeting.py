"""Build ZipCount training/val data from REAL meeting corpora (AMI-SDM +
NOTSOFAR-1 SDM) using word-aligned lhotse supervisions.

Each long recording is cut into fixed windows. A window is stored as a
single-"track" recipe item (source_offset into the big wav), so
SpeakerCountDataset streams it with sf.read(start, frames) — no audio is
copied to disk. Labels = per-speaker word-interval union summed, @100 Hz,
clipped to 3 (same definition as eval_bench, so train and eval agree).

Silence-only windows are KEPT (capped by --max-silence-frac of items): real
far-field room tone is exactly the class-0 material LibriMix lacks.

Usage (on the machine that has the audio):
    PYTHONPATH=. python src/data/prepare_meeting.py \
        --out-dir data/meetings \
        --ami-manifest-dir data/manifests/ami \
        --nsf-manifest-dir data/manifests/notsofar1
    mkdir -p data/combined
    cat data/librimix/train_manifest.json data/meetings/train_manifest.json \
        > data/combined/train_manifest.json
"""
import argparse
import gzip
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np

from src.data.label_utils import sups_to_frame_counts

FRAME_HZ = 100
NSF_SNAPSHOT = "240825.1"


def build_sets(ami_dir, nsf_dir, nsf_snapshot):
    return {
        # name: (recordings manifest, supervisions manifest)
        "ami_train": (os.path.join(ami_dir, "ami-sdm_recordings_train.jsonl.gz"),
                      os.path.join(ami_dir, "ami-sdm_supervisions_train.jsonl.gz")),
        "ami_dev":   (os.path.join(ami_dir, "ami-sdm_recordings_dev.jsonl.gz"),
                      os.path.join(ami_dir, "ami-sdm_supervisions_dev.jsonl.gz")),
        "nsf_train": (os.path.join(nsf_dir, f"notsofar1_sdm_train_set_{nsf_snapshot}_train_recordings.jsonl.gz"),
                      os.path.join(nsf_dir, f"notsofar1_sdm_train_set_{nsf_snapshot}_train_supervisions.jsonl.gz")),
    }


def manifests_exist(name, rec_manifest, sup_manifest):
    missing = [p for p in (rec_manifest, sup_manifest) if not os.path.exists(p)]
    if missing:
        print(f"[{name}] missing manifests, skip:")
        for p in missing:
            print(f"  MISSING {p}")
        return False
    return True


def read_jsonl_gz(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def windows_for_set(name, rec_manifest, sup_manifest, labels_dir, win, min_win, limit):
    recs = {r["id"]: (r["sources"][0]["source"], float(r["duration"]))
            for r in read_jsonl_gz(rec_manifest)}
    sups = defaultdict(list)
    n_al = n_sup = 0
    for s in read_jsonl_gz(sup_manifest):
        sups[s["recording_id"]].append(s)
        n_sup += 1
        n_al += bool((s.get("alignment") or {}).get("word"))
    print(f"[{name}] {len(recs)} recordings, {n_sup} sups "
          f"({n_al / max(n_sup, 1):.0%} word-aligned)")

    items, cls = [], Counter()
    rec_ids = sorted(recs)[:limit] if limit else sorted(recs)
    for rid in rec_ids:
        path, dur = recs[rid]
        if not os.path.exists(path):
            print(f"  MISSING audio, skip {rid}")
            continue
        counts = sups_to_frame_counts(sups.get(rid, []), dur, FRAME_HZ)
        n_win = int(dur // win) + (1 if dur % win >= min_win else 0)
        for w in range(n_win):
            s0 = w * win
            wdur = min(win, dur - s0)
            lab = counts[int(s0 * FRAME_HZ): int(s0 * FRAME_HZ) + int(wdur * FRAME_HZ)]
            item_id = f"{name}_{rid}_w{w:04d}"
            lp = os.path.join(labels_dir, item_id + ".npy")
            np.save(lp, lab.astype(np.int32))
            items.append({
                "id": item_id,
                "duration": round(wdur, 3),
                "n_spk": int(lab.max()),
                "source": name,
                "sample_rate": 16000,
                "label_filepath": os.path.abspath(lp),
                "tracks": [{"source": path, "volume": 1.0, "offset": 0.0,
                            "duration": round(wdur, 3), "source_offset": round(s0, 3)}],
            })
            for k, v in zip(*np.unique(lab, return_counts=True)):
                cls[int(k)] += int(v)
    return items, cls


def cap_silence(items, max_frac, seed=17):
    """Keep all speech windows; cap all-silence windows to max_frac of total."""
    speech = [it for it in items if it["n_spk"] > 0]
    silence = [it for it in items if it["n_spk"] == 0]
    random.Random(seed).shuffle(silence)
    keep = int(max_frac * len(speech) / max(1e-9, 1 - max_frac))
    return speech + silence[:keep], len(silence) - min(keep, len(silence))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/meetings")
    ap.add_argument("--ami-manifest-dir", default="data/manifests/ami")
    ap.add_argument("--nsf-manifest-dir", default="data/manifests/notsofar1")
    ap.add_argument("--nsf-snapshot", default=NSF_SNAPSHOT)
    ap.add_argument("--include", default="ami,nsf",
                    help="Comma-separated corpora to include in train: ami,nsf")
    ap.add_argument("--win", type=float, default=30.0)
    ap.add_argument("--min-win", type=float, default=5.0, help="min tail window length")
    ap.add_argument("--max-silence-frac", type=float, default=0.10)
    ap.add_argument("--limit", type=int, default=0, help="recordings per set (smoke)")
    args = ap.parse_args()

    include = {x.strip().lower() for x in args.include.split(",") if x.strip()}
    train_sets = []
    if "ami" in include:
        train_sets.append("ami_train")
    if "nsf" in include:
        train_sets.append("nsf_train")

    sets = build_sets(args.ami_manifest_dir, args.nsf_manifest_dir, args.nsf_snapshot)
    for split, set_names in [("train", train_sets), ("val", ["ami_dev"])]:
        labels_dir = os.path.join(args.out_dir, f"labels_{split}")
        os.makedirs(labels_dir, exist_ok=True)
        all_items, cls = [], Counter()
        for name in set_names:
            rec_m, sup_m = sets[name]
            if not manifests_exist(name, rec_m, sup_m):
                continue
            items, c = windows_for_set(name, rec_m, sup_m, labels_dir,
                                       args.win, args.min_win, args.limit)
            all_items += items
            cls.update(c)
        if not all_items:
            raise RuntimeError(f"No {split} items were created; check manifest paths.")
        all_items, n_dropped = cap_silence(all_items, args.max_silence_frac)
        random.Random(17).shuffle(all_items)
        mpath = os.path.join(args.out_dir, f"{split}_manifest.json")
        with open(mpath, "w") as f:
            for it in all_items:
                f.write(json.dumps(it) + "\n")
        tot = sum(cls.values())
        dist = {k: round(cls[k] / tot, 4) for k in sorted(cls)}
        hours = sum(it["duration"] for it in all_items) / 3600
        print(f"[{split}] {len(all_items)} windows ({hours:.1f}h, "
              f"dropped {n_dropped} extra silence windows) -> {mpath}")
        print(f"[{split}] frame dist (before silence cap): {dist}")

    print("done. Combine with librimix:")
    print("  mkdir -p data/combined && cat data/librimix/train_manifest.json "
          f"{args.out_dir}/train_manifest.json > data/combined/train_manifest.json")


if __name__ == "__main__":
    main()
