"""Head-to-head: NVIDIA Sortformer-STREAMING (diar_streaming_sortformer_4spk-v2.1)
on the SAME unified protocol as the existing 3-way comparison (DiariZen 0.765,
ZipCount 0.649, Sortformer-offline 0.606 — see zipcount-eval-protocol memory).

Motivation: comparing streaming ZipCount against OFFLINE Sortformer (unbounded
context) is not latency-fair. This script reuses:
  - the SAME windowed assets as the unified comparison
    (data_ext/nemo_eval/ami_test/{manifest.json,wav,rttm}, built by
    make_nemo_eval_assets.py — the exact files NVIDIA's official
    e2e_diarize_speech.py scored to produce the 0.606 offline number),
  - the SAME ground-truth function (rttm_to_frame_counts),
  - the SAME prediction rasterization / scoring (segments_to_counts,
    f1_from_conf, binary_f1, imported from eval_sortformer_ami),
  - the SAME 570-window common-window restriction directory
    (logs/diarizen_ami_test/pred_rttms; DiariZen fails 13 of the 583 AMI-SDM
    test windows).

Only the model and its streaming latency knobs differ. `--chunk-len` /
`--chunk-right-context` (in SUBSAMPLED frames; this checkpoint: subsampling
factor 8 @ 10 ms/frame = 80 ms/subsampled-frame) override the checkpoint's
own defaults (188 / 1 -> ~15.1 s algorithmic latency) so a genuinely
latency-matched row can be produced (8 / 0 -> exactly 640 ms, matching
ZipCount's own 16-output-frame/640 ms causal cadence with zero lookahead).
Shrinking below the trained operating point is OFF-LABEL: report it as such,
not as NVIDIA's recommended setting.

Example (default streaming, ~15s):
  PYTHONPATH=. python src/eval_sortformer_streaming.py \
      --model-path /home/edabk/MinhDuc/zipformer-vi/nemo/model/diar_streaming_sortformer_4spk-v2.1.nemo \
      --manifest data_ext/nemo_eval/ami_test/manifest.json \
      --restrict-to-dir logs/diarizen_ami_test/pred_rttms \
      --name "streaming-default"

Example (latency-matched, 640 ms, off-label):
  ... --chunk-len 8 --chunk-right-context 0 --name "streaming-640ms"
"""
import argparse

import numpy as np
import json

from src.data.label_utils import rttm_to_frame_counts
from src.eval_sortformer_ami import binary_f1, f1_from_conf, segments_to_counts

SUBSAMPLING_FACTOR = 8
WINDOW_STRIDE_S = 0.01  # preprocessor mel-frame hop


def write_rttm(path, uid, segments):
    """[begin_s, end_s, speaker_idx] triples -> standard RTTM lines."""
    with open(path, "w") as f:
        for seg in segments:
            if isinstance(seg, (list, tuple)):
                s, e, spk = float(seg[0]), float(seg[1]), seg[2]
            else:
                parts = str(seg).split()
                s, e, spk = float(parts[0]), float(parts[1]), parts[2]
            dur = e - s
            if dur > 0:
                f.write(
                    f"SPEAKER {uid} 1 {s:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n"
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True,
                    help="local .nemo streaming checkpoint (restore_from, not from_pretrained)")
    ap.add_argument("--manifest", default="data_ext/nemo_eval/ami_test/manifest.json")
    ap.add_argument("--restrict-to-dir", default="logs/diarizen_ami_test/pred_rttms",
                    help="common-window set every system in the 3-way table used")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name", default="streaming")
    ap.add_argument("--postprocessing-yaml", default=None)
    ap.add_argument("--chunk-len", type=int, default=None,
                    help="override; None = use the checkpoint's own trained default")
    ap.add_argument("--chunk-right-context", type=int, default=None)
    ap.add_argument("--spkcache-update-period", type=int, default=None,
                    help="default: match --chunk-len if either latency knob is overridden")
    ap.add_argument("--dump-rttm-dir", default=None,
                    help="also write predicted RTTMs here (for scripts/score_der.py)")
    args = ap.parse_args()

    import os
    if args.dump_rttm_dir:
        os.makedirs(args.dump_rttm_dir, exist_ok=True)
    restrict = None
    if args.restrict_to_dir:
        restrict = {os.path.splitext(f)[0] for f in os.listdir(args.restrict_to_dir)
                    if f.endswith(".rttm")}

    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.restore_from(args.model_path, map_location="cuda")
    model.eval()
    if not model.streaming_mode:
        raise SystemExit(
            f"{args.model_path} does not have streaming_mode=True; this "
            "script is only meaningful for a streaming checkpoint"
        )
    sm = model.sortformer_modules
    if args.chunk_len is not None:
        sm.chunk_len = args.chunk_len
    if args.chunk_right_context is not None:
        sm.chunk_right_context = args.chunk_right_context
    if args.spkcache_update_period is not None:
        sm.spkcache_update_period = args.spkcache_update_period
    elif args.chunk_len is not None or args.chunk_right_context is not None:
        # keep "update every chunk" semantics under an overridden chunk size,
        # instead of silently falling back with a logged warning
        sm.spkcache_update_period = sm.chunk_len
    sm._check_streaming_parameters()

    latency_ms = (sm.chunk_len + sm.chunk_right_context) * SUBSAMPLING_FACTOR * WINDOW_STRIDE_S * 1000
    print(f"Loaded {args.model_path}")
    print(
        f"streaming params: chunk_len={sm.chunk_len} chunk_right_context={sm.chunk_right_context} "
        f"spkcache_len={sm.spkcache_len} fifo_len={sm.fifo_len} "
        f"spkcache_update_period={sm.spkcache_update_period}"
    )
    print(f"algorithmic latency: {latency_ms:.0f} ms "
          f"(chunk_len + chunk_right_context, in subsampled frames @ "
          f"{SUBSAMPLING_FACTOR * WINDOW_STRIDE_S * 1000:.0f} ms/frame)")

    items = [json.loads(line) for line in open(args.manifest)]
    if restrict is not None:
        items = [it for it in items
                 if os.path.splitext(os.path.basename(it["audio_filepath"]))[0] in restrict]
    if args.limit:
        items = items[:args.limit]
    print(f"{args.name}: scoring {len(items)} windows"
          f"{' (restricted to the common set)' if restrict else ''}")

    conf = np.zeros((4, 4), dtype=np.int64)
    for i in range(0, len(items), args.batch_size):
        chunk = items[i:i + args.batch_size]
        wavs = [it["audio_filepath"] for it in chunk]
        segs = model.diarize(audio=wavs, batch_size=len(wavs), verbose=False,
                             postprocessing_yaml=args.postprocessing_yaml)
        for j, it in enumerate(chunk):
            gt = rttm_to_frame_counts(it["rttm_filepath"], it["duration"])
            pred = segments_to_counts(segs[j], len(gt))
            np.add.at(conf, (gt, pred), 1)
            if args.dump_rttm_dir:
                uid = os.path.splitext(os.path.basename(it["audio_filepath"]))[0]
                write_rttm(
                    os.path.join(args.dump_rttm_dir, uid + ".rttm"), uid, segs[j]
                )
        if (i // args.batch_size) % 20 == 0:
            print(f"  [{i + len(chunk)}/{len(items)}]", flush=True)

    acc = np.trace(conf) / conf.sum()
    idx = np.indices(conf.shape)
    mae = float((np.abs(idx[0] - idx[1]) * conf).sum() / conf.sum())
    f1s = [f1_from_conf(conf, c)[0] for c in range(4)]
    gt_flat = np.repeat(idx[0].ravel(), conf.ravel())
    pr_flat = np.repeat(idx[1].ravel(), conf.ravel())
    vad_f1, vad_p, vad_r = binary_f1((gt_flat >= 1).astype(int), (pr_flat >= 1).astype(int))
    osd_f1, osd_p, osd_r = binary_f1((gt_flat >= 2).astype(int), (pr_flat >= 2).astype(int))

    print(f"\n===== Sortformer-{args.name} (streaming, {latency_ms:.0f} ms) — "
          f"unified protocol ({len(items)} windows, 100 Hz) =====")
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
