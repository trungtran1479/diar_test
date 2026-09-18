"""Score ZipCount on the SAME windows, at the SAME frame rate, as the official
competitor runs.

Why this exists: the three systems were previously scored under three different
protocols and the numbers were compared anyway.

  ZipCount    lhotse AMI-SDM, 30 s windows, 25 Hz (the model's own output rate)
  Sortformer  NeMo eval assets, 90 s windows, 100 Hz
  DiariZen    NeMo eval assets, 90 s windows, 100 Hz

Window length changes how much context an offline system gets, and frame rate
changes how boundary frames are counted, so those macro-F1 values were never
comparable. This evaluator puts ZipCount on the competitors' protocol: the same
NeMo manifest, the same ground-truth RTTMs rasterised by the same function, and
predictions upsampled from the model's 25 Hz to 100 Hz by nearest-neighbour
repeat — which is exactly what a deployed streaming system would emit.

Upsampling cannot invent precision, and that is the point: ZipCount genuinely
has 40 ms resolution, so scoring it at 100 Hz charges it honestly for boundary
frames it cannot place.

Use --restrict-to-dir to score only windows every system produced output for
(DiariZen fails on 13 short windows), so no system is credited for skipping.
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
import yaml

from src.data.feature_extractor import feature_extractor_for_config
from src.data.label_utils import rttm_to_frame_counts
from src.eval_sortformer_ami import binary_f1, f1_from_conf
from src.models.zipcount_v1 import build_model

N_CLS = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--restrict-to-dir", default=None,
                    help="score only uids that have an RTTM here (common-window set)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--name", default="ZipCount")
    ap.add_argument("--dump-conf-json", default=None,
                    help="also write {uid: 4x4 confusion} here, for "
                         "scripts/paired_bootstrap.py")
    args = ap.parse_args()

    restrict = None
    if args.restrict_to_dir:
        restrict = {os.path.splitext(f)[0] for f in os.listdir(args.restrict_to_dir)
                    if f.endswith(".rttm")}

    config = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    feature_fn = feature_extractor_for_config(config)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck))
    model = model.to(device).eval()

    items = [json.loads(l) for l in open(args.manifest)]
    if restrict is not None:
        items = [it for it in items
                 if os.path.splitext(os.path.basename(it["audio_filepath"]))[0] in restrict]
    print(f"{args.name}: scoring {len(items)} windows "
          f"{'(restricted to the common set)' if restrict else '(all windows)'}")

    conf = np.zeros((N_CLS, N_CLS), dtype=np.int64)
    per_uid = {}
    ratio_seen = set()
    with torch.no_grad():
        for i in range(0, len(items), args.batch_size):
            chunk = items[i:i + args.batch_size]
            feats, flens, gts = [], [], []
            for it in chunk:
                wav, sr = sf.read(it["audio_filepath"], dtype="float32")
                if wav.ndim > 1:
                    wav = wav[:, 0]
                # returns (num_frames, 80) already — NOT batched, so no [0]
                # (or (num_samples,) for a raw-waveform extractor, e.g. WavLM)
                f = feature_fn(torch.from_numpy(wav).unsqueeze(0), sr)
                feats.append(f)
                flens.append(f.shape[0])
                gts.append(rttm_to_frame_counts(it["rttm_filepath"], it["duration"]))
            # pad_sequence (not a manual torch.zeros(..., feats[0].shape[-1])
            # buffer) so this works for BOTH [T, 80] frame features and 1-D
            # [T] raw-waveform features — shape[-1] on a 1-D tensor is T
            # itself, which silently built a [B, T, T] buffer and crashed
            # the assignment below for any waveform-based backbone.
            batch = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True, padding_value=0.0)
            logits, h_lens = model(batch.to(device),
                                   torch.tensor(flens, device=device))
            pred25 = logits[0].float().argmax(-1).cpu().numpy()

            for j, it in enumerate(chunk):
                n_h = int(h_lens[j])
                gt = gts[j]
                # model rate -> 100 Hz. The factor is derived per batch rather
                # than hard-coded, so a config change to the downsampling
                # schedule cannot silently misalign the two sequences.
                up = max(1, int(round(len(gt) / max(n_h, 1))))
                ratio_seen.add(up)
                p = np.repeat(pred25[j, :n_h], up)
                n = min(len(p), len(gt))
                np.add.at(conf, (gt[:n], p[:n]), 1)
                if args.dump_conf_json:
                    cw = np.zeros((N_CLS, N_CLS), dtype=np.int64)
                    np.add.at(cw, (gt[:n], p[:n]), 1)
                    uid = os.path.splitext(os.path.basename(it["audio_filepath"]))[0]
                    per_uid[uid] = cw.ravel().tolist()

    if args.dump_conf_json:
        with open(args.dump_conf_json, "w") as fh:
            json.dump(per_uid, fh)
        print(f"wrote per-window confusions for {len(per_uid)} windows "
              f"-> {args.dump_conf_json}")
    if len(ratio_seen) > 1:
        print(f"WARNING: inconsistent upsample factors across batches: {sorted(ratio_seen)}")
    gt_tot = conf.sum()
    acc = np.trace(conf) / gt_tot
    f1s = [f1_from_conf(conf, c)[0] for c in range(N_CLS)]
    idx = np.indices(conf.shape)
    mae = float((np.abs(idx[0] - idx[1]) * conf).sum() / gt_tot)
    gt_flat = np.repeat(idx[0].ravel(), conf.ravel())
    pr_flat = np.repeat(idx[1].ravel(), conf.ravel())
    vad = binary_f1((gt_flat >= 1).astype(int), (pr_flat >= 1).astype(int))
    osd = binary_f1((gt_flat >= 2).astype(int), (pr_flat >= 2).astype(int))

    print(f"\n===== {args.name} — same protocol as competitors "
          f"({len(items)} windows, 100 Hz, upsample x{sorted(ratio_seen)}) =====")
    print(f"acc={acc:.3f}  mae={mae:.3f}  macroF1={np.mean(f1s):.3f}")
    print("f1/cls = [" + " ".join(f"{f:.3f}" for f in f1s) + "]")
    print(f"VAD:  F1={vad[0]:.3f} (P={vad[1]:.3f} R={vad[2]:.3f})")
    print(f"OSD (count>=2): F1={osd[0]:.3f} (P={osd[1]:.3f} R={osd[2]:.3f})")
    print("confusion P(pred|true):  pred0   pred1   pred2   pred3")
    for t in range(N_CLS):
        row = conf[t] / max(conf[t].sum(), 1)
        print(f"  true{t}:               " + "   ".join(f"{v:.3f}" for v in row))


if __name__ == "__main__":
    main()
