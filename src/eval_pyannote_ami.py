"""pyannote.audio (speaker-diarization-3.1) on the SAME unified protocol as
the other systems in the AMI-SDM comparison table (DiariZen 0.765, ZipCount
0.649, Sortformer-offline 0.606, Sortformer-streaming default/640ms).

pyannote/speaker-diarization-3.1 is the most commonly-cited open diarization
baseline in this literature; its absence would be noticed by reviewers.
Like Sortformer-offline, this is an OFFLINE pipeline (full-utterance
clustering) -- label it as such, it is a reference row, not a streaming
comparison.

Run in the `pyannote_eval` conda env (isolated from `.zipformer`: pyannote.audio
4.x requires torch>=2.8, which would have upgraded/broken the zipformer/k2
CUDA pipeline if installed into `.zipformer`).

Example:
  /home/edabk/miniconda3/envs/pyannote_eval/bin/python -m src.eval_pyannote_ami \
      --manifest data_ext/nemo_eval/ami_test/manifest.json \
      --restrict-to-dir logs/diarizen_ami_test/pred_rttms \
      --dump-rttm-dir logs/pyannote_ami_test/pred_rttms
"""
import argparse
import json
import os

import numpy as np
import torch

from src.data.label_utils import rttm_to_frame_counts
from src.eval_sortformer_ami import binary_f1, f1_from_conf, segments_to_counts


def annotation_to_segments(annotation):
    """pyannote Annotation -> [(start, end, speaker), ...], same shape
    eval_sortformer_ami.segments_to_counts already expects."""
    return [
        (segment.start, segment.end, label)
        for segment, _, label in annotation.itertracks(yield_label=True)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data_ext/nemo_eval/ami_test/manifest.json")
    ap.add_argument("--restrict-to-dir", default="logs/diarizen_ami_test/pred_rttms")
    ap.add_argument("--dump-rttm-dir", default="logs/pyannote_ami_test/pred_rttms")
    ap.add_argument("--pipeline-name", default="pyannote/speaker-diarization-3.1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name", default="pyannote-3.1")
    args = ap.parse_args()

    restrict = None
    if args.restrict_to_dir:
        restrict = {os.path.splitext(f)[0] for f in os.listdir(args.restrict_to_dir)
                    if f.endswith(".rttm")}

    # This env (pyannote_eval) pulls torch 2.13+cu13 / nvidia-cudnn-cu13
    # 9.20.0.48, which fails EVERY cuDNN conv on this machine with
    # CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH (reproduced with a bare
    # conv1d, unrelated to pyannote) -- a broken bleeding-edge cudnn
    # install/driver combo, not a code bug. Falling back to the non-cuDNN
    # conv path keeps this on GPU (just without cuDNN-accelerated kernels).
    torch.backends.cudnn.enabled = False

    token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(args.pipeline_name, token=token)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    print(f"Loaded {args.pipeline_name} on {device} (cudnn disabled, see comment)")

    if args.dump_rttm_dir:
        os.makedirs(args.dump_rttm_dir, exist_ok=True)

    items = [json.loads(line) for line in open(args.manifest)]
    if restrict is not None:
        items = [it for it in items
                 if os.path.splitext(os.path.basename(it["audio_filepath"]))[0] in restrict]
    if args.limit:
        items = items[:args.limit]
    print(f"{args.name}: scoring {len(items)} windows"
          f"{' (restricted to the common set)' if restrict else ''}")

    conf = np.zeros((4, 4), dtype=np.int64)
    for i, it in enumerate(items):
        uid = os.path.splitext(os.path.basename(it["audio_filepath"]))[0]
        # .speaker_diarization keeps overlapping speech turns (needed for
        # count>=2 frames); .exclusive_speaker_diarization deliberately
        # drops overlap for downstream transcription and would undercount.
        annotation = pipeline(it["audio_filepath"]).speaker_diarization
        if args.dump_rttm_dir:
            with open(os.path.join(args.dump_rttm_dir, uid + ".rttm"), "w") as f:
                annotation.write_rttm(f)

        gt = rttm_to_frame_counts(it["rttm_filepath"], it["duration"])
        pred = segments_to_counts(annotation_to_segments(annotation), len(gt))
        np.add.at(conf, (gt, pred), 1)
        if i % 20 == 0:
            print(f"  [{i + 1}/{len(items)}] {uid}", flush=True)

    acc = np.trace(conf) / conf.sum()
    idx = np.indices(conf.shape)
    mae = float((np.abs(idx[0] - idx[1]) * conf).sum() / conf.sum())
    f1s = [f1_from_conf(conf, c)[0] for c in range(4)]
    gt_flat = np.repeat(idx[0].ravel(), conf.ravel())
    pr_flat = np.repeat(idx[1].ravel(), conf.ravel())
    vad_f1, vad_p, vad_r = binary_f1((gt_flat >= 1).astype(int), (pr_flat >= 1).astype(int))
    osd_f1, osd_p, osd_r = binary_f1((gt_flat >= 2).astype(int), (pr_flat >= 2).astype(int))

    print(f"\n===== {args.name} (offline) — unified protocol "
          f"({len(items)} windows, 100 Hz) =====")
    print(f"acc={acc:.3f}  mae={mae:.3f}  macroF1={np.mean(f1s):.3f}")
    print("f1/cls = [" + " ".join(f"{f:.3f}" for f in f1s) + "]")
    print(f"VAD:  F1={vad_f1:.3f} (P={vad_p:.3f} R={vad_r:.3f})")
    print(f"OSD (count>=2): F1={osd_f1:.3f} (P={osd_p:.3f} R={osd_r:.3f})")
    print("confusion P(pred|true):  pred0   pred1   pred2   pred3")
    for t in range(4):
        row = conf[t] / max(conf[t].sum(), 1)
        print(f"  true{t}:               " + "   ".join(f"{v:.3f}" for v in row))


if __name__ == "__main__":
    main()
