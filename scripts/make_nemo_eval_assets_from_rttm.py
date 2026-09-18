"""Build NeMo-format eval assets (window wavs + ground-truth RTTMs + manifest)
from a SINGLE COMBINED RTTM file + a directory of per-recording wavs.

Sibling of `make_nemo_eval_assets.py`, which requires lhotse recordings/
supervisions `.jsonl.gz` manifests (only available for ami/dipco/notsofar1
under data/manifests/). VoxConverse and MSDWild don't have those, but their
ORIGINAL per-speaker RTTM ground truth exists on disk as one combined file
per split -- this script windows from that instead. Output schema (wav/
rttm/ manifest.json, 90 s windows) is IDENTICAL, so every downstream script
(eval_sortformer_streaming.py, eval_pyannote_ami.py, eval_zipcount_nemo.py,
run_diarizen.py, NVIDIA's official e2e_diarize_speech.py, score_der.py)
works against it unmodified.

Note: this project's own window-JSON label .npy files (used for ZipCount
training) are derived COUNT-only labels, not speaker-attributed -- they
cannot be reversed into RTTM. The original corpus RTTMs are the only
correct ground-truth source here.

Examples:
  PYTHONPATH=. python scripts/make_nemo_eval_assets_from_rttm.py \
      --rttm "/media/edabk/500GB Hard Disk/data_diar/voxconverse/rttm/test.rttm" \
      --audio-dir "/media/edabk/500GB Hard Disk/data_diar/voxconverse/audio_test/voxconverse_test_wav" \
      --out data_ext/nemo_eval/voxconverse_test

  PYTHONPATH=. python scripts/make_nemo_eval_assets_from_rttm.py \
      --rttm "/media/edabk/500GB Hard Disk/data_diar/msdwild/rttm/many.val.rttm" \
      --audio-dir "/media/edabk/500GB Hard Disk/data_diar/msdwild/audio/wav" \
      --restrict-ids-from "/media/edabk/500GB Hard Disk/data_diar/msdwild/msdwild_manyval_LOCKED.json" \
      --restrict-id-prefix msdwild_manyval_ \
      --out data_ext/nemo_eval/msdwild_manyval
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import soundfile as sf

FRAME_HZ = 100


def read_rttm(path):
    """RTTM -> {recording_id: [(start, end, speaker), ...]}."""
    segments = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                rid = parts[1]
                start, dur, spk = float(parts[3]), float(parts[4]), parts[7]
                if dur > 0:
                    segments[rid].append((start, start + dur, spk))
    return segments


def per_speaker_tracks(segments, duration):
    """{speaker: bool array @100Hz} -- same shape make_nemo_eval_assets.py uses."""
    nf = int(round(duration * FRAME_HZ))
    per_spk = {}
    for start, end, spk in segments:
        mask = per_spk.setdefault(spk, np.zeros(nf, dtype=bool))
        i, j = max(0, int(round(start * FRAME_HZ))), min(nf, int(round(end * FRAME_HZ)))
        if i < j:
            mask[i:j] = True
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


def load_restrict_ids(path, prefix):
    ids = set()
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            rid = item["id"].rsplit("_w", 1)[0]
            if prefix and rid.startswith(prefix):
                rid = rid[len(prefix):]
            ids.add(rid)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rttm", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--win", type=float, default=90.0)
    ap.add_argument("--audio-ext", default=".wav")
    ap.add_argument("--restrict-ids-from", default=None,
                    help="window-JSON manifest; only build assets for "
                         "recordings whose (prefix-stripped) id appears here")
    ap.add_argument("--restrict-id-prefix", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    all_segments = read_rttm(args.rttm)
    rec_ids = sorted(all_segments)
    if args.restrict_ids_from:
        keep = load_restrict_ids(args.restrict_ids_from, args.restrict_id_prefix)
        missing = keep - set(rec_ids)
        if missing:
            raise SystemExit(
                f"{len(missing)} restricted ids not found in RTTM, e.g. "
                f"{sorted(missing)[:5]}"
            )
        rec_ids = [r for r in rec_ids if r in keep]
    if args.limit:
        rec_ids = rec_ids[:args.limit]

    wav_dir = os.path.join(args.out, "wav")
    rttm_dir = os.path.join(args.out, "rttm")
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(rttm_dir, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")

    n_win = 0
    with open(manifest_path, "w") as mf:
        for ri, rid in enumerate(rec_ids):
            path = os.path.join(args.audio_dir, rid + args.audio_ext)
            if not os.path.isfile(path):
                raise SystemExit(f"missing audio for recording {rid}: {path}")
            info = sf.info(path)
            sr = info.samplerate
            dur = info.frames / sr
            tracks = per_speaker_tracks(all_segments[rid], dur)
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
                f0 = int(s0 * FRAME_HZ)
                f1 = f0 + int(win_dur * FRAME_HZ)
                with open(rttm_path, "w") as rf:
                    for spk, mask in tracks.items():
                        for a, b in runs(mask[f0:f1]):
                            rf.write(
                                f"SPEAKER {uid} 1 {a / FRAME_HZ:.3f} "
                                f"{(b - a) / FRAME_HZ:.3f} <NA> <NA> {spk} <NA> <NA>\n"
                            )
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
    print(f"Wrote {n_win} windows over {len(rec_ids)} recordings -> {manifest_path}")


if __name__ == "__main__":
    main()
