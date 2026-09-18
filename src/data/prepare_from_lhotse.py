"""Build clean ZipCount training data from lhotse LibriMix MixedCut cutsets.

Input cuts are on-the-fly mixtures: tracks = (source flac, volume, offset,
duration). Raw labels from offsets alone are dirty:
  - no class-0 frames at all (tracks cover the whole cut),
  - every intra-utterance pause is labeled "speaking".

Cleaning performed here:
  1. Per-SOURCE energy VAD (sources are separate files, so this is exact,
     no separation needed) -> speech intervals per track -> summed counts.
  2. Integrity filter: drop cuts with missing/unreadable sources, degenerate
     durations, or (optionally) duration > --max-duration.
  3. Class balancing material: --num-solo single-speaker items sampled from
     the track sources (class 1/0) and --num-noise MUSAN/silence items
     (class 0).
  4. Speaker-disjoint split by construction: train cutsets vs dev cutsets
     (verified: 0 speaker overlap).

Output manifest lines keep the ORIGINAL mixing recipe instead of audio paths:
    {"id", "duration", "n_spk", "label_filepath", "source", "sample_rate",
     "tracks": [{"source", "volume", "offset", "duration"}, ...]}
so SpeakerCountDataset mixes on the fly (no need to render ~440 h of wavs).

Usage (full run, ~130k unique sources VAD-ed in parallel, cached):
    PYTHONPATH=. python src/data/prepare_from_lhotse.py \
        --out-dir data/librimix \
        --musan-manifest /home/pc/v2t/data/musan_lhotse/musan_recordings_noise.jsonl.gz \
        --num-solo 12000 --num-noise 6000 --workers 16
"""
import argparse
import gzip
import json
import os
import random
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np
import soundfile as sf

BASE = os.environ.get(
    "LIBRIMIX_CUTSET_DIR",
    "/home/pc/v2t/data/en/mtasr_prep/manifests/librimix",
)
TRAIN_SETS = [
    ("libri2mix_train-clean-100", "librimix_cutset_libri2mix_train-clean-100_30s.jsonl.gz"),
    ("libri2mix_train-clean-360", "librimix_cutset_libri2mix_train-clean-360_30s.jsonl.gz"),
    ("libri3mix_train-clean-100", "librimix_cutset_libri3mix_train-clean-100_30s.jsonl.gz"),
    ("libri3mix_train-clean-360", "librimix_cutset_libri3mix_train-clean-360_30s.jsonl.gz"),
]
VAL_SETS = [
    ("libri2mix_dev-clean", "librimix_cutset_libri2mix_dev-clean.jsonl.gz"),
    ("libri3mix_dev-clean", "librimix_cutset_libri3mix_dev-clean.jsonl.gz"),
]

FRAME_HZ = 100  # label rate


# ----------------------------------------------------------------------
# Energy VAD on a single clean source file
# ----------------------------------------------------------------------

def energy_vad_intervals(
    path: str,
    rel_db: float = 35.0,     # speech = frames within rel_db of the file's peak RMS
    abs_db: float = -55.0,    # ... but never below this absolute floor (dBFS)
    win_s: float = 0.025,
    hop_s: float = 0.01,
    close_gap_s: float = 0.20,   # bridge pauses shorter than this (they stay "speech")
    min_speech_s: float = 0.10,  # drop speech islands shorter than this
):
    """Returns (duration_s, [(start_frame, end_frame), ...]) at 100 Hz, or None on read error."""
    try:
        x, sr = sf.read(path, dtype="float32")
    except Exception:
        return None
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = len(x)
    if n == 0:
        return None
    win, hop = int(win_s * sr), int(hop_s * sr)
    if n < win:
        return (n / sr, [])

    cs = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    starts = np.arange(0, n - win + 1, hop)
    rms = np.sqrt((cs[starts + win] - cs[starts]) / win) + 1e-10
    rms_db = 20.0 * np.log10(rms)

    thr = max(rms_db.max() - rel_db, abs_db)
    mask = rms_db > thr

    # morphology on the 100 Hz grid
    def runs(m, val):
        out, s = [], None
        for i, v in enumerate(m):
            if v == val and s is None:
                s = i
            elif v != val and s is not None:
                out.append((s, i)); s = None
        if s is not None:
            out.append((s, len(m)))
        return out

    close_gap = int(close_gap_s / hop_s)
    for a, b in runs(mask, False):
        if 0 < a and b < len(mask) and (b - a) <= close_gap:
            mask[a:b] = True
    min_speech = int(min_speech_s / hop_s)
    for a, b in runs(mask, True):
        if (b - a) < min_speech:
            mask[a:b] = False

    return (n / sr, [(int(a), int(b)) for a, b in runs(mask, True)])


def _vad_worker(path):
    return path, energy_vad_intervals(path)


# ----------------------------------------------------------------------
# Cutset parsing
# ----------------------------------------------------------------------

def iter_cuts(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def cut_to_item(cut, source_name):
    """MixedCut json -> compact item dict (mixing recipe + speakers)."""
    tracks, speakers = [], []
    for t in cut["tracks"]:
        mc = t["cut"]
        rec = mc["recording"]
        vol = 1.0
        for tr in rec.get("transforms") or []:
            if tr["name"] == "Volume":
                vol = float(tr["kwargs"]["factor"])
        src = rec["sources"][0]["source"]
        for s in mc.get("supervisions", []):
            speakers.append(s.get("speaker", "?"))
        tracks.append({
            "source": src,
            "volume": vol,
            "offset": float(t["offset"]),
            "duration": float(mc["duration"]),
        })
    duration = max(tr["offset"] + tr["duration"] for tr in tracks)
    return {
        "id": f"{source_name}_{cut['id']}",
        "duration": round(duration, 3),
        "n_spk": len(tracks),
        "speakers": speakers,
        "source": source_name,
        "sample_rate": 16000,
        "tracks": tracks,
    }


# ----------------------------------------------------------------------
# Label building
# ----------------------------------------------------------------------

def build_label(item, vad_cache, use_vad=True):
    nf = int(round(item["duration"] * FRAME_HZ))
    counts = np.zeros(nf, dtype=np.int8)
    for tr in item["tracks"]:
        off = int(round(tr["offset"] * FRAME_HZ))
        dur_f = int(round(tr["duration"] * FRAME_HZ))
        vad = vad_cache.get(tr["source"]) if use_vad else None
        if vad:
            for a, b in vad[1]:
                lo, hi = min(off + a, nf), min(off + min(b, dur_f), nf)
                if lo < hi:
                    counts[lo:hi] += 1
        else:  # fallback: whole-utterance activity
            lo, hi = min(off, nf), min(off + dur_f, nf)
            if lo < hi:
                counts[lo:hi] += 1
    return np.clip(counts, 0, 3).astype(np.int32)


# ----------------------------------------------------------------------
# Extra class-balancing items
# ----------------------------------------------------------------------

def make_solo_items(all_tracks, num, vad_cache, source_name="solo_1spk", max_dur=30.0):
    """Single-speaker items sampled from the mixture sources (class 1 + 0)."""
    pool = [t for t in all_tracks if t["duration"] <= max_dur and t["source"] in vad_cache]
    random.shuffle(pool)
    items = []
    for i, tr in enumerate(pool[:num]):
        items.append({
            "id": f"{source_name}_{i:06d}",
            "duration": round(tr["duration"], 3),
            "n_spk": 1,
            "source": source_name,
            "sample_rate": 16000,
            "tracks": [{"source": tr["source"], "volume": 1.0, "offset": 0.0,
                        "duration": tr["duration"]}],
        })
    return items


def make_noise_items(musan_manifest, num, dur_range=(3.0, 20.0), p_noise=0.8):
    """Class-0 items: MUSAN noise segments (mixed on the fly) or pure silence."""
    noise = []
    if musan_manifest and os.path.exists(musan_manifest):
        opener = gzip.open if musan_manifest.endswith(".gz") else open
        with opener(musan_manifest, "rt") as f:
            for line in f:
                r = json.loads(line)
                src = r.get("sources", [{}])[0].get("source") or r.get("audio_filepath")
                dur = r.get("duration", 0)
                if src and os.path.exists(src) and dur > 1.0:
                    noise.append((src, float(dur)))
    items = []
    for i in range(num):
        d = round(random.uniform(*dur_range), 3)
        if noise and random.random() < p_noise:
            src, sdur = random.choice(noise)
            off = random.uniform(0, max(0.0, sdur - d))
            use_d = min(d, sdur)
            tracks = [{"source": src, "volume": random.uniform(0.2, 1.0),
                       "offset": 0.0, "duration": round(use_d, 3),
                       "source_offset": round(off, 3)}]
            d = round(use_d, 3)
        else:
            tracks = []  # pure silence
        items.append({
            "id": f"noise0_{i:06d}", "duration": d, "n_spk": 0,
            "source": "musan_noise" if tracks else "silence",
            "sample_rate": 16000, "tracks": tracks,
        })
    return items


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def process_split(sets, limit, args, vad_cache, labels_dir, manifest_path, split_name):
    items, cls = [], Counter()
    kept = dropped = 0
    for source_name, fn in sets:
        n_from_set = 0
        for cut in iter_cuts(os.path.join(BASE, fn)):
            if limit and n_from_set >= limit:
                break
            item = cut_to_item(cut, source_name)
            if item["duration"] <= 0 or item["duration"] > args.max_duration:
                dropped += 1
                continue
            if any(t["source"] not in vad_cache for t in item["tracks"]) and args.vad:
                dropped += 1  # unreadable source
                continue
            items.append(item)
            kept += 1
            n_from_set += 1
    print(f"[{split_name}] mixtures kept={kept} dropped={dropped}")

    if split_name == "train":
        all_tracks = [t for it in items for t in it["tracks"]]
        solos = make_solo_items(all_tracks, args.num_solo, vad_cache)
        noises = make_noise_items(args.musan_manifest, args.num_noise)
        print(f"[{split_name}] + {len(solos)} solo(1spk) + {len(noises)} noise/silence(0spk)")
        items += solos + noises
    else:
        noises = make_noise_items(args.musan_manifest, args.val_noise)
        items += noises

    random.shuffle(items)
    os.makedirs(labels_dir, exist_ok=True)
    with open(manifest_path, "w") as f:
        for item in items:
            if item["n_spk"] == 0:
                label = np.zeros(int(round(item["duration"] * FRAME_HZ)), dtype=np.int32)
            else:
                label = build_label(item, vad_cache, use_vad=args.vad)
            lp = os.path.join(labels_dir, item["id"] + ".npy")
            np.save(lp, label)
            item["label_filepath"] = os.path.abspath(lp)
            item.pop("speakers", None)
            for k, v in zip(*np.unique(label, return_counts=True)):
                cls[int(k)] += int(v)
            f.write(json.dumps(item) + "\n")
    tot = sum(cls.values())
    dist = {k: round(cls[k] / tot, 4) for k in sorted(cls)}
    print(f"[{split_name}] items={len(items)}  frame dist={dist}")
    return cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/librimix")
    ap.add_argument("--musan-manifest", default="")
    ap.add_argument("--num-solo", type=int, default=12000)
    ap.add_argument("--num-noise", type=int, default=6000)
    ap.add_argument("--val-noise", type=int, default=400)
    ap.add_argument("--max-duration", type=float, default=30.0)
    ap.add_argument("--vad", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="cuts per set (smoke test)")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = os.path.join(args.out_dir, "vad_cache.json.gz")

    # ---- collect unique sources ----
    unique = set()
    for source_name, fn in TRAIN_SETS + VAL_SETS:
        n_from_set = 0
        for cut in iter_cuts(os.path.join(BASE, fn)):
            if args.limit and n_from_set >= args.limit:
                break
            n_from_set += 1
            for t in cut["tracks"]:
                unique.add(t["cut"]["recording"]["sources"][0]["source"])
    print(f"unique source files: {len(unique)}")

    # ---- VAD with cache ----
    vad_cache = {}
    if os.path.exists(cache_path):
        with gzip.open(cache_path, "rt") as f:
            vad_cache = {k: (v[0], [tuple(iv) for iv in v[1]])
                         for k, v in json.load(f).items() if v is not None}
        print(f"loaded VAD cache: {len(vad_cache)} entries")
    todo = sorted(unique - set(vad_cache))
    if args.vad and todo:
        print(f"running energy VAD on {len(todo)} files with {args.workers} workers...")
        with Pool(args.workers) as pool:
            for i, (path, res) in enumerate(pool.imap_unordered(_vad_worker, todo, chunksize=64)):
                if res is not None:
                    vad_cache[path] = res
                if (i + 1) % 10000 == 0:
                    print(f"  VAD {i + 1}/{len(todo)}")
        with gzip.open(cache_path, "wt") as f:
            json.dump({k: [v[0], list(map(list, v[1]))] for k, v in vad_cache.items()}, f)
        print(f"VAD cache saved: {len(vad_cache)} entries -> {cache_path}")

    # ---- speech coverage sanity ----
    if vad_cache:
        cov = [sum(b - a for a, b in iv) / max(d * FRAME_HZ, 1)
               for d, iv in list(vad_cache.values())[:5000]]
        print(f"VAD speech coverage per source: mean={np.mean(cov):.3f} p5={np.percentile(cov,5):.3f} "
              f"p95={np.percentile(cov,95):.3f} (LibriSpeech ~0.75-0.95 expected)")

    # ---- build splits ----
    cls_train = process_split(
        TRAIN_SETS, args.limit, args, vad_cache,
        labels_dir=os.path.join(args.out_dir, "labels_train"),
        manifest_path=os.path.join(args.out_dir, "train_manifest.json"),
        split_name="train")
    process_split(
        VAL_SETS, args.limit, args, vad_cache,
        labels_dir=os.path.join(args.out_dir, "labels_val"),
        manifest_path=os.path.join(args.out_dir, "val_manifest.json"),
        split_name="val")

    # ---- class weights from train ----
    tot = sum(cls_train.values())
    inv = [tot / max(cls_train.get(i, 0), 1) for i in range(4)]
    mean_inv = sum(inv) / 4.0
    weights = [round(min(f / mean_inv, 20.0), 4) for f in inv]
    stats = {
        "frame_counts": {i: int(cls_train.get(i, 0)) for i in range(4)},
        "class_weights": weights,
        "total_frames": int(tot),
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"class weights: {weights}")
    print("done.")


if __name__ == "__main__":
    main()
