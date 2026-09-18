"""Standard benchmarks for ZipCount — "is it competitive?"

  --bench ami         Frame-level count/VAD/OSD on AMI-SDM (real far-field
                      meetings). Labels built from lhotse supervisions using
                      WORD-level alignments when available (per-speaker union,
                      gaps <= 0.3s bridged), else full segments. Long
                      recordings are processed in fixed windows.
  --bench libricount  Clip-level speaker count on LibriCount
                      (ground truth 0..10 mapped to {0,1,2,3+}).

Examples:
  PYTHONPATH=. python src/eval_bench.py --bench ami --split dev \
      --config configs/zipcount_v2.yaml \
      --checkpoint result/r3_v2_uniform_sord/best_macro_f1.pt
  PYTHONPATH=. python src/eval_bench.py --bench libricount \
      --config configs/zipcount_v2.yaml \
      --checkpoint result/r3_v2_uniform_sord/best_macro_f1.pt
"""
import argparse
import glob
import gzip
import json
import os
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf
import torch
import yaml

from src.data.feature_extractor import feature_extractor_for_config
from src.data.label_utils import align_batch_labels, sups_to_frame_counts
from src.models.zipcount_v1 import build_model
from src.utils.metrics import compute_metrics, aggregate_metrics, format_confusion

AMI_MANIFEST_DIR = os.environ.get("AMI_MANIFEST_DIR", "data/manifests/ami")
LIBRICOUNT_DIR = os.environ.get("LIBRICOUNT_DIR", "data_ext/libricount/test")
FRAME_HZ = 100


def load_model(config_path, ckpt_path, device):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model = build_model(config)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model = model.to(device).eval()
    return model, config


def read_jsonl_gz(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


# ---------------------------------------------------------------------------
# AMI
# ---------------------------------------------------------------------------

def ami_frame_counts(sups, duration):
    return sups_to_frame_counts(sups, duration, FRAME_HZ)


def run_batch(model, feats_list, labels_list, device, use_amp=True):
    lens = torch.tensor([f.size(0) for f in feats_list], dtype=torch.long)
    feats = torch.nn.utils.rnn.pad_sequence(feats_list, batch_first=True).to(device)
    lab_lens = torch.tensor([len(l) for l in labels_list], dtype=torch.long)
    labs = torch.nn.utils.rnn.pad_sequence(
        [torch.from_numpy(l).long() for l in labels_list],
        batch_first=True, padding_value=-100)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
        logits, h_lens = model(feats, lens.to(device))
        aligned = align_batch_labels(labs.to(device), lab_lens.to(device), h_lens)
    return compute_metrics(logits[0].float(), logits[1].float(), logits[2].float(),
                           aligned, h_lens)


def bench_ami(model, args, device, extract_features):
    mdir = args.manifest_dir or AMI_MANIFEST_DIR
    recs = {}
    for r in read_jsonl_gz(os.path.join(mdir, f"{args.prefix}_recordings_{args.split}.jsonl.gz")):
        src = r["sources"][0]["source"]
        if args.source_endswith:
            src = next(s["source"] for s in r["sources"]
                       if s["source"].endswith(args.source_endswith))
        recs[r["id"]] = (src, r["duration"])
    sups = defaultdict(list)
    n_word_al = 0
    for s in read_jsonl_gz(os.path.join(mdir, f"{args.prefix}_supervisions_{args.split}.jsonl.gz")):
        sups[s["recording_id"]].append(s)
        if (s.get("alignment") or {}).get("word"):
            n_word_al += 1
    total_sups = sum(len(v) for v in sups.values())
    print(f"{args.prefix} {args.split}: {len(recs)} recordings, {total_sups} supervisions "
          f"({n_word_al} with word alignment = {n_word_al / max(total_sups,1):.0%})")

    rec_ids = sorted(recs)[:args.limit] if args.limit else sorted(recs)
    all_metrics, buf_f, buf_l = [], [], []
    win = args.win
    for ri, rid in enumerate(rec_ids):
        path, dur = recs[rid]
        counts = ami_frame_counts(sups.get(rid, []), dur)
        info = sf.info(path)
        sr = info.samplerate
        n_win = int(np.ceil(dur / win))
        for w in range(n_win):
            s0 = w * win
            n_want = int(min(win, dur - s0) * sr)
            if n_want < sr:  # skip sub-second tails
                continue
            x, _ = sf.read(path, dtype="float32", start=int(s0 * sr), frames=n_want)
            if x.ndim > 1:
                x = x.mean(axis=1)
            if args.norm == "rms":
                # Far-field AMI is much quieter than the close-talk training
                # audio; match loudness to LibriSpeech-ish level (~-26 dBFS).
                rms = float(np.sqrt(np.mean(x ** 2)) + 1e-8)
                x = np.clip(x * (0.05 / rms), -1.0, 1.0)
            elif args.norm == "peak":
                x = x * (0.7 / (np.abs(x).max() + 1e-8))
            feats = extract_features(torch.from_numpy(x).unsqueeze(0))
            lab = counts[int(s0 * FRAME_HZ): int(s0 * FRAME_HZ) + int(len(x) / sr * FRAME_HZ)]
            buf_f.append(feats)
            buf_l.append(lab)
            if len(buf_f) == args.batch_size:
                m = run_batch(model, buf_f, buf_l, device)
                if m:
                    all_metrics.append(m)
                buf_f, buf_l = [], []
        print(f"  [{ri + 1}/{len(rec_ids)}] {rid} ({dur / 60:.0f} min)", flush=True)
    if buf_f:
        m = run_batch(model, buf_f, buf_l, device)
        if m:
            all_metrics.append(m)

    res = aggregate_metrics(all_metrics)
    print(f"\n===== {args.prefix} {args.split} — frame-level (25 Hz) =====")
    print(f"acc={res['acc']:.3f}  mae={res['mae']:.3f}  macroF1={res['macro_f1']:.3f}")
    print(f"f1/cls = [{res['f1_0']:.3f} {res['f1_1']:.3f} {res['f1_2']:.3f} {res['f1_3']:.3f}]")
    print(f"VAD:  F1={res['f1_vad']:.3f} (P={res['p_vad']:.3f} R={res['r_vad']:.3f})")
    print(f"OSD (count>=2): F1={res['f1_ov_count']:.3f} (P={res['p_ov_count']:.3f} R={res['r_ov_count']:.3f})")
    print(f"OSD (aux head): F1={res['f1_ov_head']:.3f} (P={res['p_ov_head']:.3f} R={res['r_ov_head']:.3f})")
    print(format_confusion(res))
    return res


# ---------------------------------------------------------------------------
# LibriCount
# ---------------------------------------------------------------------------

def bench_libricount(model, args, device, extract_features):
    wavs = sorted(glob.glob(os.path.join(LIBRICOUNT_DIR, "*.wav")))
    if args.limit:
        wavs = wavs[:args.limit]
    print(f"LibriCount: {len(wavs)} clips (5 s each), gt 0..10 -> {{0,1,2,3+}}")

    conf = np.zeros((4, 4), dtype=np.int64)     # capped gt x pred(p80)
    per_k = Counter(); per_k_ok = Counter()     # by raw gt k
    mae_sum = 0.0

    for i in range(0, len(wavs), args.batch_size):
        chunk = wavs[i:i + args.batch_size]
        feats_list, gts = [], []
        for p in chunk:
            x, sr = sf.read(p, dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            feats_list.append(extract_features(torch.from_numpy(x).unsqueeze(0)))
            gts.append(int(os.path.basename(p).split("_")[0]))
        lens = torch.tensor([f.size(0) for f in feats_list], dtype=torch.long)
        feats = torch.nn.utils.rnn.pad_sequence(feats_list, batch_first=True).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits, h_lens = model(feats, lens.to(device))
        preds = logits[0].float().argmax(-1)     # [B, T]
        for b, gt in enumerate(gts):
            fp = preds[b, :h_lens[b]].cpu().numpy()
            clip_pred = int(np.percentile(fp, 80))  # robust "max simultaneous"
            gt_c = min(gt, 3)
            conf[gt_c, clip_pred] += 1
            per_k[gt] += 1
            per_k_ok[gt] += int(clip_pred == gt_c)
            mae_sum += abs(clip_pred - gt_c)

    n = conf.sum()
    acc = np.trace(conf) / n
    print(f"\n===== LibriCount (clip-level, decoder=p80) =====")
    print(f"acc={acc:.3f}  mae(capped)={mae_sum / n:.3f}   n={n}")
    print("confusion (rows=gt capped, cols=pred):")
    for i in range(4):
        row = conf[i] / max(conf[i].sum(), 1)
        print(f"  gt{i}: " + "  ".join(f"{v:.3f}" for v in row))
    print("accuracy by raw speaker count k:")
    print("  " + "  ".join(f"k={k}:{per_k_ok[k]/per_k[k]:.2f}" for k in sorted(per_k)))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bench", choices=["ami", "libricount"], required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "eval"])
    ap.add_argument("--manifest-dir", default=None,
                    help="lhotse manifest dir (default: AMI_MANIFEST_DIR)")
    ap.add_argument("--prefix", default="ami-sdm", help="manifest prefix, e.g. dipco-mdm")
    ap.add_argument("--source-endswith", default=None,
                    help="pick the recording source ending with this (multi-channel sets)")
    ap.add_argument("--win", type=float, default=30.0, help="window seconds (ami)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--norm", default="none", choices=["none", "rms", "peak"],
                    help="per-window input gain normalization (ami)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.config, args.checkpoint, device)
    extract_features = feature_extractor_for_config(config)
    print(f"Loaded {args.checkpoint} on {device}")

    if args.bench == "ami":
        bench_ami(model, args, device, extract_features)
    else:
        bench_libricount(model, args, device, extract_features)


if __name__ == "__main__":
    main()
