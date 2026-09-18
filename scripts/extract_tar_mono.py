"""Disk-safe streaming extraction: read a (possibly huge) .tar.gz of
multi-channel wavs in a SINGLE sequential pass and write each one out as
channel-0 16 kHz mono. The multi-channel original is never written to disk,
so peak extra space is one file in RAM (not the whole extracted tar).

Used for AliMeeting far (78 GB, 8-channel) where the extracted tree would not
fit on the remaining disk. Channel 0 = one distant array mic, matching how we
treat AMI-SDM (Array1-01) — a single far-field channel for streaming counting.

Example:
  PYTHONPATH=. python scripts/extract_tar_mono.py \
      --tar "/media/.../alimeeting/raw/Train_Ali_far.tar.gz" \
      --out "/media/.../alimeeting/mono" \
      --member-glob "*.wav" --strip-suffix .wav
"""
import argparse
import fnmatch
import io
import os
import tarfile

import numpy as np
import soundfile as sf
import torchaudio
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--member-glob", default="*.wav")
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--target-sr", type=int, default=16000)
    ap.add_argument("--strip-suffix", default="")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n = 0
    resamplers = {}
    with tarfile.open(args.tar, "r|gz") as tf:   # streaming, sequential
        for m in tf:
            if not m.isfile() or not fnmatch.fnmatch(os.path.basename(m.name), args.member_glob):
                continue
            base = os.path.basename(m.name)
            if args.strip_suffix and base.endswith(args.strip_suffix):
                base = base[: -len(args.strip_suffix)]
            out_path = os.path.join(args.out, base + ".wav")
            if os.path.exists(out_path):
                n += 1
                continue
            buf = io.BytesIO(tf.extractfile(m).read())   # one file in RAM
            x, sr = sf.read(buf, dtype="float32")
            if x.ndim > 1:
                x = x[:, args.channel]
            if sr != args.target_sr:
                if sr not in resamplers:
                    resamplers[sr] = torchaudio.transforms.Resample(sr, args.target_sr)
                x = resamplers[sr](torch.from_numpy(np.ascontiguousarray(x))).numpy()
            sf.write(out_path, x, args.target_sr)
            n += 1
            if n % 20 == 0:
                print(f"  {n} files -> {args.out}", flush=True)
    print(f"Done. {n} mono files in {args.out}")


if __name__ == "__main__":
    main()
