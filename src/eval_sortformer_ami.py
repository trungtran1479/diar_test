"""Head-to-head: NVIDIA Sortformer (diar_sortformer_4spk-v1, OFFLINE) on the
same AMI-SDM frame-count protocol as `src/eval_bench.py --bench ami`.

Ground truth is built with the exact same code path (sups_to_frame_counts,
word alignments + gap bridging, 100 Hz). Sortformer's decoded segments are
rasterized onto the same 100 Hz grid; frame count = number of simultaneously
active predicted speakers, capped at 3+.

Caveats for a fair reading: Sortformer v1 is an offline model (bidirectional
attention over the whole window) while ZipCount is causal streaming; it is
also trained on far more diarization data. It sees the audio in --win second
windows here (default 90 s, its training context length).

Example:
  PYTHONPATH=. python src/eval_sortformer_ami.py --split test \
      > logs/sortformer_eval_ami_test.log 2>&1
"""
import argparse
import gzip
import json
import os
import shutil
import tempfile
from collections import defaultdict

import numpy as np
import soundfile as sf

from src.data.label_utils import sups_to_frame_counts

AMI_MANIFEST_DIR = os.environ.get("AMI_MANIFEST_DIR", "data/manifests/ami")
FRAME_HZ = 100


def read_jsonl_gz(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def segments_to_counts(segments, n_frames):
    """Rasterize decoded segments (['<start> <end> <speaker>', ...]) to a
    100 Hz speaker-count track, capped at 3+."""
    cnt = np.zeros(n_frames, dtype=np.int64)
    for seg in segments:
        if isinstance(seg, (list, tuple)):
            s, e = float(seg[0]), float(seg[1])
        else:
            parts = str(seg).split()
            s, e = float(parts[0]), float(parts[1])
        i0 = max(int(round(s * FRAME_HZ)), 0)
        i1 = min(int(round(e * FRAME_HZ)), n_frames)
        if i0 < i1:
            cnt[i0:i1] += 1
    return np.minimum(cnt, 3)


def f1_from_conf(conf, cls):
    tp = conf[cls, cls]
    fp = conf[:, cls].sum() - tp
    fn = conf[cls, :].sum() - tp
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return f1, p, r


def binary_f1(gt, pred):
    tp = int(((gt == 1) & (pred == 1)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-9), p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--win", type=float, default=90.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="nvidia/diar_sortformer_4spk-v1")
    ap.add_argument("--postprocessing-yaml", default=None,
                    help="NeMo postprocessing YAML (e.g. NVIDIA's DIHARD3-dev "
                         "optimized params); default = NeMo built-in defaults")
    args = ap.parse_args()

    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.from_pretrained(args.model, map_location="cuda")
    model.eval()
    print(f"Loaded {args.model}")

    recs = {r["id"]: (r["sources"][0]["source"], r["duration"])
            for r in read_jsonl_gz(os.path.join(
                AMI_MANIFEST_DIR, f"ami-sdm_recordings_{args.split}.jsonl.gz"))}
    sups = defaultdict(list)
    for s in read_jsonl_gz(os.path.join(
            AMI_MANIFEST_DIR, f"ami-sdm_supervisions_{args.split}.jsonl.gz")):
        sups[s["recording_id"]].append(s)
    rec_ids = sorted(recs)[:args.limit] if args.limit else sorted(recs)
    print(f"AMI-SDM {args.split}: {len(rec_ids)} recordings, win={args.win}s")

    tmpdir = tempfile.mkdtemp(prefix="sortformer_ami_")
    gt_all, pred_all = [], []
    try:
        for ri, rid in enumerate(rec_ids):
            path, dur = recs[rid]
            counts = sups_to_frame_counts(sups.get(rid, []), dur, FRAME_HZ)
            info = sf.info(path)
            sr = info.samplerate
            wav_paths, gts = [], []
            for w in range(int(np.ceil(dur / args.win))):
                s0 = w * args.win
                n_want = int(min(args.win, dur - s0) * sr)
                if n_want < sr:
                    continue
                x, _ = sf.read(path, dtype="float32", start=int(s0 * sr), frames=n_want)
                if x.ndim > 1:
                    x = x.mean(axis=1)
                p = os.path.join(tmpdir, f"{rid}_{w:04d}.wav")
                sf.write(p, x, sr)
                wav_paths.append(p)
                g0 = int(s0 * FRAME_HZ)
                gts.append(counts[g0: g0 + int(len(x) / sr * FRAME_HZ)])
            for i in range(0, len(wav_paths), args.batch_size):
                chunk = wav_paths[i:i + args.batch_size]
                segs = model.diarize(audio=chunk, batch_size=len(chunk), verbose=False,
                                     postprocessing_yaml=args.postprocessing_yaml)
                for b in range(len(chunk)):
                    gt = np.minimum(gts[i + b], 3)
                    pred = segments_to_counts(segs[b], len(gt))
                    gt_all.append(gt)
                    pred_all.append(pred)
            for p in wav_paths:
                os.remove(p)
            print(f"  [{ri + 1}/{len(rec_ids)}] {rid} ({dur / 60:.0f} min)", flush=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    gt = np.concatenate(gt_all)
    pred = np.concatenate(pred_all)
    conf = np.zeros((4, 4), dtype=np.int64)
    np.add.at(conf, (gt, pred), 1)

    acc = np.trace(conf) / conf.sum()
    mae = np.abs(gt - pred).mean()
    f1s = [f1_from_conf(conf, c)[0] for c in range(4)]
    vad_f1, vad_p, vad_r = binary_f1((gt >= 1).astype(int), (pred >= 1).astype(int))
    osd_f1, osd_p, osd_r = binary_f1((gt >= 2).astype(int), (pred >= 2).astype(int))

    print(f"\n===== Sortformer 4spk-v1 (offline) — AMI-SDM {args.split}, "
          f"frame-level ({FRAME_HZ} Hz) =====")
    print(f"acc={acc:.3f}  mae={mae:.3f}  macroF1={np.mean(f1s):.3f}")
    print(f"f1/cls = [" + " ".join(f"{f:.3f}" for f in f1s) + "]")
    print(f"VAD:  F1={vad_f1:.3f} (P={vad_p:.3f} R={vad_r:.3f})")
    print(f"OSD (count>=2): F1={osd_f1:.3f} (P={osd_p:.3f} R={osd_r:.3f})")
    print("confusion P(pred|true):  pred0   pred1   pred2   pred3")
    for t in range(4):
        row = conf[t] / max(conf[t].sum(), 1)
        print(f"  true{t}:               " + "   ".join(f"{v:.3f}" for v in row))


if __name__ == "__main__":
    main()
