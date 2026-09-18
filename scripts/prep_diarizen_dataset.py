"""Turn a DiariZen-format diarization dataset (RTTM + wav.scp, or RTTM + an
audio dir) into ZipCount training windows: 30 s windowed manifest items +
100 Hz frame-count label .npy (0/1/2/3+), tagged with a `source` for
oversampling/augmentation control.

Reuses the exact ZipCount label convention (rttm_to_frame_counts @100 Hz,
capped at 3) and the on-the-fly single-track mixing path (tracks=[{source,
offset, duration, source_offset}]) so no audio is duplicated on disk — the
manifest points into the original recording at a source_offset.

Examples:
  # DiariZen ships wav.scp (id -> wav path) + one combined rttm:
  PYTHONPATH=. python scripts/prep_diarizen_dataset.py \
      --rttm third_party/DiariZen/recipes/diar_ssl/data/AMI_AliMeeting_AISHELL4/train/rttm \
      --wav-scp third_party/DiariZen/recipes/diar_ssl/data/AMI_AliMeeting_AISHELL4/train/wav.scp \
      --source ami_train --only-prefix EN,ES,IS,TS \
      --out-manifest data/combined/ami_dz_train.json \
      --out-label-dir data/meetings/labels_dz_ami_train

  # or a plain audio dir (VoxConverse/MSDWild/RAMC after you download them):
  PYTHONPATH=. python scripts/prep_diarizen_dataset.py \
      --rttm data_ext/voxconverse/train.rttm \
      --audio-dir data_ext/voxconverse/audio --audio-ext .wav \
      --source voxconverse_train \
      --out-manifest data/combined/voxconverse_train.json \
      --out-label-dir data/meetings/labels_voxconverse_train
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import soundfile as sf

from src.data.label_utils import rttm_to_frame_counts  # noqa: F401  (parity)

FRAME_HZ = 100


def parse_rttm(path):
    """recording_id -> list of (start, dur, speaker)."""
    segs = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 8 and p[0] == "SPEAKER":
                segs[p[1]].append((float(p[3]), float(p[4]), p[7]))
    return segs


def frame_counts(segments, n_frames):
    """Instantaneous speaker count @100 Hz, capped at 3 (per-speaker union so
    overlapping turns of the SAME speaker are not double-counted)."""
    per_spk = {}
    for s, d, spk in segments:
        m = per_spk.setdefault(spk, np.zeros(n_frames, dtype=bool))
        i = max(0, int(round(s * FRAME_HZ)))
        j = min(n_frames, int(round((s + d) * FRAME_HZ)))
        if i < j:
            m[i:j] = True
    cnt = np.zeros(n_frames, dtype=np.int32)
    for m in per_spk.values():
        cnt += m.astype(np.int32)
    return np.clip(cnt, 0, 3)


def resolve_audio(rid, wav_scp, audio_dir, audio_ext):
    if wav_scp is not None:
        return wav_scp.get(rid)
    cand = os.path.join(audio_dir, rid + audio_ext)
    return cand if os.path.exists(cand) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rttm", required=True)
    ap.add_argument("--wav-scp", default=None)
    ap.add_argument("--audio-dir", default=None)
    ap.add_argument("--audio-ext", default=".wav")
    ap.add_argument("--source", required=True, help="source tag, e.g. alimeeting_train")
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--out-label-dir", required=True)
    ap.add_argument("--win", type=float, default=30.0)
    ap.add_argument("--only-prefix", default=None,
                    help="comma list; keep only recording ids starting with one "
                         "of these (to split the combined AMI/Ali/AISHELL rttm)")
    ap.add_argument("--min-win", type=float, default=5.0,
                    help="drop trailing windows shorter than this")
    args = ap.parse_args()

    wav_scp = None
    if args.wav_scp:
        wav_scp = {}
        for line in open(args.wav_scp):
            # split once only: audio paths may contain spaces
            # (e.g. "/media/edabk/500GB Hard Disk/...")
            parts = line.rstrip("\n").split(None, 1)
            if len(parts) >= 2:
                wav_scp[parts[0]] = parts[1].strip()
    prefixes = tuple(args.only_prefix.split(",")) if args.only_prefix else None

    os.makedirs(args.out_label_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_manifest)), exist_ok=True)

    segs_by_rec = parse_rttm(args.rttm)
    n_rec = n_win = n_missing = 0
    import json
    with open(args.out_manifest, "w") as mf:
        for rid in sorted(segs_by_rec):
            if prefixes and not rid.startswith(prefixes):
                continue
            wav = resolve_audio(rid, wav_scp, args.audio_dir, args.audio_ext)
            if not wav or not os.path.exists(wav):
                n_missing += 1
                continue
            info = sf.info(wav)
            dur = info.frames / info.samplerate
            counts = frame_counts(segs_by_rec[rid], int(round(dur * FRAME_HZ)))
            n_rec += 1
            n_full = int(dur // args.win)
            n_windows = int(np.ceil(dur / args.win))
            for w in range(n_windows):
                s0 = w * args.win
                wdur = min(args.win, dur - s0)
                if wdur < args.min_win:
                    continue
                f0 = int(round(s0 * FRAME_HZ))
                lab = counts[f0: f0 + int(round(wdur * FRAME_HZ))]
                uid = f"{args.source}_{rid}_w{w:04d}"
                lab_path = os.path.abspath(os.path.join(args.out_label_dir, uid + ".npy"))
                np.save(lab_path, lab)
                mf.write(json.dumps({
                    "id": uid,
                    "duration": round(wdur, 3),
                    "n_spk": int(lab.max()),
                    "source": args.source,
                    "sample_rate": 16000,
                    "label_filepath": lab_path,
                    "tracks": [{
                        "source": wav,
                        "volume": 1.0,
                        "offset": 0.0,
                        "duration": round(wdur, 3),
                        "source_offset": round(s0, 3),
                    }],
                }) + "\n")
                n_win += 1
    print(f"{args.source}: {n_rec} recordings -> {n_win} windows "
          f"({n_missing} recordings had no audio, skipped)")
    print(f"  manifest: {args.out_manifest}")
    print(f"  labels:   {args.out_label_dir}")


if __name__ == "__main__":
    main()
