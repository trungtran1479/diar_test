"""Per-RECORDING sequence metrics that flattened confusion counts cannot see.

Companion to ``eval_per_recording.py``: that script pools every window's
frames into one confusion matrix per recording, which is correct for F1/MAE
but destroys frame ORDER, so it cannot see over-segmentation (flickering
predictions) or calibration. This script stitches a recording's fixed-length
windows back together in order (ids end in ``_w####``; consecutive windows of
a recording are contiguous, non-overlapping — verified against the manifests
this project uses) and computes:

  - segmental edit score      (chatgpt_recommend.txt section 11)
  - fragmentation rate        (section 11)
  - transition confusion      (DOWN/STABLE/UP count-change events, section 11)
  - expected calibration error + Brier score, pooled over the whole corpus

Count MAE and boundary F1/AP already exist elsewhere (``metrics.py``'s
``aggregate_metrics`` and ``eval_boundary_ap.py`` respectively) and are not
duplicated here.

Example:
  PYTHONPATH=. python scripts/eval_sequence_metrics.py \
      --manifest .../vox_sel.json --checkpoint logs/.../step3000.pt \
      --config artifacts/.../s1234.yaml --out results/seq_s1.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.dataset import SpeakerCountDataset, collate_fn
from src.data.feature_extractor import feature_extractor_for_config
from src.data.label_utils import align_batch_labels
from src.models.zipcount_v1 import build_model
from src.utils.sequence_metrics import (
    aggregate_sequence_metrics,
    brier_score,
    count_mae,
    expected_calibration_error,
    fragmentation_rate,
    segmental_edit_score,
    transition_confusion,
)

_WINDOW_RE = re.compile(r"_w(\d+)$")


def rec_of(window_id: str) -> str:
    return window_id.rsplit("_w", 1)[0]


def window_index(window_id: str) -> int:
    match = _WINDOW_RE.search(window_id)
    if not match:
        raise ValueError(f"window id has no _w#### suffix: {window_id!r}")
    return int(match.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ece-bins", type=int, default=15)
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
    ids = [ds.data[i]["id"] for i in range(len(ds))]
    order = sorted(range(len(ids)), key=lambda i: (rec_of(ids[i]), window_index(ids[i])))
    # Batching does not disturb ordering: Subset(ds, order) is already sorted
    # by (recording, window index), and shuffle=False + batching just groups
    # CONSECUTIVE elements of that fixed order -- per-recording buffers below
    # accumulate correctly regardless of batch size.
    loader = DataLoader(
        Subset(ds, order), batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4,
    )

    buffers: dict = defaultdict(lambda: {"true": [], "pred": []})
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            logits, h_lens = model(
                batch["features"].to(device), batch["feature_lens"].to(device)
            )
            aligned = align_batch_labels(
                batch["labels"].to(device), batch["label_lens"].to(device), h_lens
            )
            probs = torch.softmax(logits[0].float(), dim=-1)
            pred = probs.argmax(-1)
            for i, wid in enumerate(batch["ids"]):
                n = int(h_lens[i])
                true_w = aligned[i, :n].cpu().numpy()
                pred_w = pred[i, :n].cpu().numpy()
                probs_w = probs[i, :n].cpu().numpy()
                valid = true_w >= 0
                rid = rec_of(wid)
                buffers[rid]["true"].extend(true_w[valid].tolist())
                buffers[rid]["pred"].extend(pred_w[valid].tolist())
                if valid.any():
                    all_probs.append(probs_w[valid])
                    all_labels.append(true_w[valid])

    per_recording = {}
    for rid, seqs in buffers.items():
        true_seq, pred_seq = seqs["true"], seqs["pred"]
        per_recording[rid] = {
            "n_frames": len(true_seq),
            "edit_score": segmental_edit_score(true_seq, pred_seq),
            "fragmentation_rate": fragmentation_rate(true_seq, pred_seq),
            "transition_confusion": transition_confusion(true_seq, pred_seq),
        }

    pooled_probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 4))
    pooled_labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,), dtype=np.int64)

    out = {
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "n_recordings": len(per_recording),
        "n_frames": int(pooled_labels.shape[0]),
        "calibration": {
            "ece": expected_calibration_error(
                pooled_probs, pooled_labels, n_bins=args.ece_bins
            ),
            "brier": brier_score(pooled_probs, pooled_labels),
            "mae": count_mae(pooled_probs, pooled_labels),
        },
        "aggregate": aggregate_sequence_metrics(per_recording),
        "per_recording": per_recording,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    agg = out["aggregate"]
    print(
        f"{os.path.basename(args.checkpoint)}: "
        f"edit={agg['mean_edit_score']:.2f} "
        f"frag={agg['mean_fragmentation_rate']:.3f} "
        f"ece={out['calibration']['ece']:.4f} "
        f"brier={out['calibration']['brier']:.4f} "
        f"mae={out['calibration']['mae']:.4f} "
        f"| {len(per_recording)} recordings -> {args.out}"
    )


if __name__ == "__main__":
    main()
