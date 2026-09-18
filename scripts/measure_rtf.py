"""Measure REAL (not just algorithmic) streaming latency: per-chunk
wall-clock time and RTF (real-time factor) of the exact production
streaming inference path.

Reuses the same chunk loop as src/infer_streaming.py --
backbone.forward_streaming + model.head.forward_streaming with carried
states (KV/conv caches) -- because that IS the deployment code path; timing
the offline forward() would misrepresent real latency. The 640ms figure
used everywhere else in this project is the ALGORITHMIC chunk size (16
output frames @ 25 Hz); this script measures whether the compute to
produce one such chunk actually fits inside that budget.

Usage:
  PYTHONPATH=. python scripts/measure_rtf.py \
      --config artifacts/stack_mask_graduation/stackgrad_pre012_s1234.yaml \
      --checkpoint logs/stackgrad_pre012_s1234/step3000.pt \
      --audio-dir "/media/edabk/500GB Hard Disk/data_diar/voxconverse" \
      --num-files 8 --device cuda

  ... --device cpu --threads 1     # honest single-thread CPU number
  ... --device cpu                 # default (all cores) CPU number
"""
import argparse
import glob
import math
import os
import time

import numpy as np
import torch
import yaml

from src.data.feature_extractor import extract_fbank_features, load_audio_mono_16k
from src.models.zipcount_v1 import build_model

LOG_EPS = math.log(1e-10)


def find_wavs(audio_dir, num_files):
    wavs = sorted(glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True))
    if not wavs:
        raise SystemExit(f"no .wav files found under {audio_dir}")
    return wavs[:num_files]


def measure_one(model, backbone, wav_path, device, warmup_chunks, record=True):
    t0 = time.perf_counter()
    waveform = load_audio_mono_16k(wav_path)
    audio_duration = waveform.shape[1] / 16000
    features = extract_fbank_features(waveform, sample_rate=16000)
    if device.type == "cuda":
        torch.cuda.synchronize()
    feat_time = time.perf_counter() - t0

    T = features.size(0)
    chunk_size = backbone.chunk_size
    pad_length = backbone.pad_length
    hop = 2 * chunk_size
    win = hop + pad_length
    features = torch.nn.functional.pad(
        features, (0, 0, 0, pad_length + hop), mode="constant", value=LOG_EPS
    ).to(device)

    states = backbone.get_init_states(batch_size=1, device=device)
    head_caches = None
    num_steps = math.ceil(T / hop)

    chunk_times = []
    with torch.no_grad():
        for step in range(num_steps):
            start = step * hop
            chunk = features[start:start + win].unsqueeze(0)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            h, h_lens, states = backbone.forward_streaming(chunk, states)
            if hasattr(model.head, "forward_streaming"):
                _, head_caches = model.head.forward_streaming(h, head_caches)
            else:
                model.head(h)

            if device.type == "cuda":
                torch.cuda.synchronize()
            chunk_times.append(time.perf_counter() - t_start)

    if not record:
        return None
    chunk_times = np.array(chunk_times[warmup_chunks:]) * 1000.0  # ms
    return {
        "audio_duration": audio_duration,
        "feat_time_s": feat_time,
        "compute_time_s": chunk_times.sum() / 1000.0,
        "n_chunks": len(chunk_times),
        "chunk_ms": chunk_times,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--num-files", type=int, default=8)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch.set_num_threads, CPU only; omit = torch default")
    ap.add_argument("--warmup-chunks", type=int, default=5,
                    help="per-file chunks discarded before timing")
    args = ap.parse_args()

    if args.device == "cpu" and args.threads is not None:
        torch.set_num_threads(args.threads)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if config["model"]["encoder"].get("type") != "zipformer":
        raise SystemExit("measure_rtf.py requires encoder type 'zipformer'")

    model = build_model(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model = model.to(device).eval()
    backbone = model.backbone
    if not backbone.is_causal:
        raise SystemExit("encoder is not causal; set model.encoder.causal: true")

    wavs = find_wavs(args.audio_dir, args.num_files)
    threads_label = f"{args.threads}-thread" if args.threads else "default-thread"
    device_label = "GPU" if device.type == "cuda" else threads_label
    print(f"device={args.device} ({device_label}) files={len(wavs)} "
          f"warmup_chunks/file={args.warmup_chunks}")

    # Dedicated priming pass on the first file: absorbs one-time CUDA
    # context/cuDNN-autotune (or CPU allocator warmup) cost entirely
    # outside the measured loop, not merely the first few chunks of it.
    measure_one(model, backbone, wavs[0], device, args.warmup_chunks, record=False)

    all_chunk_ms = []
    total_audio_s = total_compute_s = total_feat_s = 0.0
    for wav in wavs:
        r = measure_one(model, backbone, wav, device, args.warmup_chunks)
        all_chunk_ms.append(r["chunk_ms"])
        total_audio_s += r["audio_duration"]
        total_compute_s += r["compute_time_s"]
        total_feat_s += r["feat_time_s"]
        print(f"  {os.path.basename(wav):40s} dur={r['audio_duration']:6.1f}s "
              f"chunks={r['n_chunks']:4d} compute={r['compute_time_s']*1000:7.1f}ms "
              f"feat={r['feat_time_s']*1000:6.1f}ms")

    chunk_ms = np.concatenate(all_chunk_ms)
    rtf = total_compute_s / total_audio_s
    print(f"\n===== RTF summary: device={args.device} ({device_label}) =====")
    print(f"total audio: {total_audio_s:.1f}s over {len(wavs)} files")
    print(f"per-chunk model compute (640ms cadence, n={len(chunk_ms)}, warmup excluded):")
    print(f"  mean={chunk_ms.mean():.2f}ms  p50={np.percentile(chunk_ms, 50):.2f}ms  "
          f"p95={np.percentile(chunk_ms, 95):.2f}ms  max={chunk_ms.max():.2f}ms")
    print(f"feature extraction: {total_feat_s * 1000 / len(wavs):.1f}ms avg/file "
          f"(computed once per file here, NOT per-chunk; a deployed system "
          f"would extract it incrementally -- reported separately, not "
          f"folded into RTF, so the two costs stay legible)")
    print(f"RTF (model compute only) = {rtf:.4f} "
          f"({'faster' if rtf < 1 else 'SLOWER'} than real-time)")
    print(f"chunk cadence budget: 640ms | mean headroom = "
          f"{640 - chunk_ms.mean():.1f}ms | p95 headroom = "
          f"{640 - np.percentile(chunk_ms, 95):.1f}ms")


if __name__ == "__main__":
    main()
