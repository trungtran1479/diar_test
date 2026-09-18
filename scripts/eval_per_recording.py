"""Per-RECORDING evaluation, so a difference between two models can be tested
against recording-level variability instead of being read off a single pooled
number.

Pooled frame-level F1 hides two things we need: whether a delta is spread across
the corpus or driven by a few long meetings, and how much of it is within the
noise of which recordings happen to be in the set. This writes one row per
recording so `compare_runs.py` can do a paired bootstrap over recordings.

Example:
  PYTHONPATH=. python scripts/eval_per_recording.py \
      --manifest .../vox_sel.json --checkpoint logs/.../step400.pt \
      --config configs/zipcount_v4_distill.yaml --out results/kd_s1.json
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import yaml

from src.data.dataset import SpeakerCountDataset, collate_fn
from src.data.feature_extractor import feature_extractor_for_config
from src.data.label_utils import align_batch_labels
from src.models.zipcount_v1 import build_model

N_CLS = 4


def rec_of(window_id: str) -> str:
    """window id -> recording id (ids end in _w####)."""
    return window_id.rsplit("_w", 1)[0]


def f1_from_conf(c, k):
    tp = c[k, k]; fp = c[:, k].sum() - tp; fn = c[k, :].sum() - tp
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return float(2 * p * r / max(p + r, 1e-9))


def summarise(conf):
    f1s = [f1_from_conf(conf, k) for k in range(N_CLS)]
    ov_tp = conf[2:, 2:].sum(); ov_fp = conf[:2, 2:].sum(); ov_fn = conf[2:, :2].sum()
    p = ov_tp / max(ov_tp + ov_fp, 1); r = ov_tp / max(ov_tp + ov_fn, 1)
    return {
        "macro_f1": float(np.mean(f1s)),
        "f1_0": f1s[0], "f1_1": f1s[1], "f1_2": f1s[2], "f1_3": f1s[3],
        "osd_f1": float(2 * p * r / max(p + r, 1e-9)),
        "acc": float(np.trace(conf) / max(conf.sum(), 1)),
        "frames": int(conf.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    config["data"]["val_manifest"] = args.manifest
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck))
    model = model.to(device).eval()

    ds = SpeakerCountDataset(args.manifest, feature_extractor=feature_extractor_for_config(config))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         collate_fn=collate_fn, num_workers=4)
    per_rec = defaultdict(lambda: np.zeros((N_CLS, N_CLS), dtype=np.int64))
    with torch.no_grad():
        for batch in loader:
            logits, h_lens = model(batch["features"].to(device),
                                   batch["feature_lens"].to(device))
            aligned = align_batch_labels(batch["labels"].to(device),
                                         batch["label_lens"].to(device), h_lens)
            pred = logits[0].float().argmax(-1)
            for i, wid in enumerate(batch["ids"]):
                n = int(h_lens[i])
                p = pred[i, :n].cpu().numpy(); l = aligned[i, :n].cpu().numpy()
                m = l >= 0
                np.add.at(per_rec[rec_of(wid)], (l[m], p[m]), 1)

    rows = {rid: summarise(c) for rid, c in per_rec.items()}
    pooled = summarise(sum(per_rec.values()))
    # unweighted mean over recordings: every meeting counts once, so one long
    # session cannot carry the result
    macro_unw = float(np.mean([r["macro_f1"] for r in rows.values()]))
    out = {
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "pooled": pooled,
        "macro_f1_unweighted_over_recordings": macro_unw,
        "n_recordings": len(rows),
        "per_recording": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"{os.path.basename(args.checkpoint)}: pooled macro={pooled['macro_f1']:.4f} "
          f"osd={pooled['osd_f1']:.4f} | unweighted-over-rec macro={macro_unw:.4f} "
          f"| {len(rows)} recordings -> {args.out}")


if __name__ == "__main__":
    main()
