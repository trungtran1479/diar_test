"""Build the r8 training manifest: the 6 real DiariZen corpora (523 h, on the
external disk) mixed with the useful parts of the old LibriMix-heavy manifest.

Mix rationale (from the r5->r7 diagnosis):
  * class 3+ was the bottleneck (F1 0.33) and it needs DATA, not loss weights.
    AliMeeting is the only real corpus with a meaningful 3+ share (6% of frames
    vs AMI's 1%) -> oversampled x3.
  * LibriMix dominated the old mix at 78% and caused the synthetic->far-field
    domain gap, but libri3mix is still the densest 3+ source (17% of frames),
    so keep a capped amount of it (far-field aug in the config makes it usable)
    instead of dropping it entirely.
  * RAMC is 2-speaker telephone with ~0% overlap: no 3+ value, so subsample.
  * ami_dz (segment labels) replaces the old ami_train (word-aligned labels):
    same recordings, so keeping both would put two label conventions on the
    same audio.

Usage:
  PYTHONPATH=. python scripts/build_r8_manifest.py \
      --out data/combined/train_manifest_r8.json
"""
import argparse
import json
import os
import random

DZ = "/media/edabk/500GB Hard Disk/data_diar/manifests"
BASE = "data/combined/train_manifest.json"

# source -> weight. >1 duplicates lines, <1 subsamples.
NEW_WEIGHTS = {
    "alimeeting_train": 3.0,   # 6% frames are 3+ speakers: the goldmine
    "aishell4_train": 1.0,
    "ami_dz_train": 1.0,
    "msdwild_train": 1.0,
    "voxconverse_train": 1.0,
    "ramc_train": 0.33,        # ~0% overlap, low value for counting
}
BASE_WEIGHTS = {
    "nsf_train": 2.0,          # real far-field, target domain
    "solo_1spk": 1.0,
    "silence": 1.0,
    "musan_noise": 1.0,
    "libri3mix_train-clean-360": 0.30,   # densest 3+ source, but capped
    "libri3mix_train-clean-100": 0.30,
    "libri2mix_train-clean-360": 0.12,   # 2-spk synthetic: least needed
    "libri2mix_train-clean-100": 0.12,
    "ami_train": 0.0,          # superseded by ami_dz_train (label convention)
}


def take(items, w, rng):
    if w >= 1:
        out = items * int(w)
        frac = w - int(w)
        if frac > 0:
            out += rng.sample(items, int(len(items) * frac))
        return out
    return rng.sample(items, int(len(items) * w)) if w > 0 else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    buckets = {}
    for name in NEW_WEIGHTS:
        path = os.path.join(DZ, f"{name}.json")
        buckets[name] = [l.rstrip("\n") for l in open(path)]
    base_by_src = {}
    for line in open(BASE):
        src = json.loads(line).get("source", "(none)")
        base_by_src.setdefault(src, []).append(line.rstrip("\n"))

    out_lines = []
    print(f"{'source':<28}{'have':>8}{'weight':>8}{'used':>8}")
    print("-" * 52)
    for name, w in NEW_WEIGHTS.items():
        got = take(buckets[name], w, rng)
        out_lines += got
        print(f"{name:<28}{len(buckets[name]):>8}{w:>8.2f}{len(got):>8}")
    for name, w in BASE_WEIGHTS.items():
        items = base_by_src.get(name, [])
        got = take(items, w, rng) if items else []
        out_lines += got
        print(f"{name:<28}{len(items):>8}{w:>8.2f}{len(got):>8}")

    rng.shuffle(out_lines)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print("-" * 52)
    print(f"{'TOTAL':<28}{'':>8}{'':>8}{len(out_lines):>8}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
