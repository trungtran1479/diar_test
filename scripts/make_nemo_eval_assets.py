"""Build NeMo-format eval assets (window wavs + ground-truth RTTMs + manifest)
from lhotse recordings/supervisions manifests, so NVIDIA's OFFICIAL
`e2e_diarize_speech.py` can be run unmodified.

Ground-truth protocol is identical to src/eval_bench.py: per-speaker activity
union, word alignments when available (gaps <= 0.3 s bridged), else full
segments. Long recordings are cut into fixed windows (default 90 s, the
Sortformer training session length) because the offline model cannot fit
30-50 min sessions in 24 GB VRAM.

Examples:
  PYTHONPATH=. python scripts/make_nemo_eval_assets.py \
      --manifest-dir data/manifests/ami --prefix ami-sdm --split test \
      --out data_ext/nemo_eval/ami_test
  PYTHONPATH=. python scripts/make_nemo_eval_assets.py \
      --manifest-dir data/manifests/dipco_mdm --prefix dipco-mdm --split eval \
      --source-endswith U01.CH1.wav --out data_ext/nemo_eval/dipco_eval
"""
import argparse
import gzip
import json
import os
from collections import defaultdict

import numpy as np
import soundfile as sf

from src.data.label_utils import sup_speech_intervals

FRAME_HZ = 100


def read_jsonl_gz(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def per_speaker_tracks(sups, duration):
    """{speaker: bool array @100Hz} using the shared gt protocol."""
    nf = int(round(duration * FRAME_HZ))
    per_spk = {}
    for sup in sups:
        m = per_spk.setdefault(sup["speaker"], np.zeros(nf, dtype=bool))
        for a, b in sup_speech_intervals(sup):
            i, j = max(0, int(round(a * FRAME_HZ))), min(nf, int(round(b * FRAME_HZ)))
            if i < j:
                m[i:j] = True
    return per_spk


def runs(mask):
    """Boolean mask -> list of (start_idx, end_idx) runs."""
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask)]
    return list(zip(starts, ends))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--prefix", required=True, help="e.g. ami-sdm, dipco-mdm")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--win", type=float, default=90.0)
    ap.add_argument("--source-endswith", default=None,
                    help="pick the source whose path ends with this "
                         "(default: first source)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    recs = {}
    for r in read_jsonl_gz(os.path.join(
            args.manifest_dir, f"{args.prefix}_recordings_{args.split}.jsonl.gz")):
        src = r["sources"][0]["source"]
        if args.source_endswith:
            matches = [s["source"] for s in r["sources"]
                       if s["source"].endswith(args.source_endswith)]
            if not matches:
                raise SystemExit(f"{r['id']}: no source ends with {args.source_endswith}")
            src = matches[0]
        recs[r["id"]] = (src, r["duration"])
    sups = defaultdict(list)
    for s in read_jsonl_gz(os.path.join(
            args.manifest_dir, f"{args.prefix}_supervisions_{args.split}.jsonl.gz")):
        sups[s["recording_id"]].append(s)

    wav_dir = os.path.join(args.out, "wav")
    rttm_dir = os.path.join(args.out, "rttm")
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(rttm_dir, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")

    rec_ids = sorted(recs)[:args.limit] if args.limit else sorted(recs)
    n_win = 0
    with open(manifest_path, "w") as mf:
        for ri, rid in enumerate(rec_ids):
            path, dur = recs[rid]
            tracks = per_speaker_tracks(sups.get(rid, []), dur)
            info = sf.info(path)
            sr = info.samplerate
            for w in range(int(np.ceil(dur / args.win))):
                s0 = w * args.win
                n_want = int(min(args.win, dur - s0) * sr)
                if n_want < sr:
                    continue
                x, _ = sf.read(path, dtype="float32", start=int(s0 * sr), frames=n_want)
                if x.ndim > 1:
                    x = x.mean(axis=1)
                uid = f"{rid}_{w:04d}"
                wav_path = os.path.abspath(os.path.join(wav_dir, uid + ".wav"))
                rttm_path = os.path.abspath(os.path.join(rttm_dir, uid + ".rttm"))
                sf.write(wav_path, x, sr)
                win_dur = len(x) / sr
                f0, f1 = int(s0 * FRAME_HZ), int(s0 * FRAME_HZ) + int(win_dur * FRAME_HZ)
                with open(rttm_path, "w") as rf:
                    for spk, m in tracks.items():
                        for a, b in runs(m[f0:f1]):
                            rf.write(f"SPEAKER {uid} 1 {a / FRAME_HZ:.3f} "
                                     f"{(b - a) / FRAME_HZ:.3f} <NA> <NA> {spk} <NA> <NA>\n")
                mf.write(json.dumps({
                    "audio_filepath": wav_path,
                    "offset": 0.0,
                    "duration": round(win_dur, 3),
                    "label": "infer",
                    "text": "-",
                    "num_speakers": None,
                    "rttm_filepath": rttm_path,
                    "uem_filepath": None,
                }) + "\n")
                n_win += 1
            print(f"[{ri + 1}/{len(rec_ids)}] {rid} ({dur / 60:.0f} min)", flush=True)
    print(f"Wrote {n_win} windows -> {manifest_path}")


if __name__ == "__main__":
    main()
