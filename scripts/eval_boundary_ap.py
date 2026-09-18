"""Boundary metrics for a pyramid-head checkpoint: tie-aware AP and recall at
a fixed false-positive budget.

Preregistered gates served here (roadmap): boundary supervision graduates only
if boundary AP >= 0.05 AND transition recall at a fixed false-transition rate
improves >= +0.05 over the carried system.

Scoring notes (second review):
  * RAW logits are ranked, never sigmoid(logits) — under fp16-trained heads
    sigmoid saturates and manufactures ties that make the metric depend on
    manifest order.  Non-finite logits are a hard error, not a quiet skip.
  * AP is computed threshold-wise (grouped by unique score), so tied scores
    contribute atomically and the result is permutation-invariant.
  * The false-positive budget is preregistered as #FP == #true boundaries
    (i.e. false-transition rate equal to boundary prevalence).  Recall is read
    at the largest threshold group whose cumulative FP stays within budget.
  * The checkpoint sha is embedded so a cached report can never speak for a
    regenerated checkpoint.

Targets follow structured_losses.build_valid_masks exactly: positive at t
means labels[t-1] != labels[t] with both frames valid.
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml

from src.data.dataset import SpeakerCountDataset, collate_fn
from src.data.feature_extractor import feature_extractor_for_config
from src.data.label_utils import align_batch_labels
from src.models.structured_losses import build_valid_masks
from src.models.zipcount_v1 import build_model


def _sha_of_file(path: str, chunk: int = 1 << 22) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def grouped_pr(scores: np.ndarray, targets: np.ndarray):
    """Cumulative tp/fp at unique-score group boundaries, descending."""
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    t = targets[order]
    # group ends: last index of each unique score run
    boundary = np.flatnonzero(np.diff(s) != 0)
    ends = np.r_[boundary, len(s) - 1]
    tp = np.cumsum(t)[ends]
    fp = np.cumsum(1 - t)[ends]
    return tp.astype(np.float64), fp.astype(np.float64)


def tie_aware_ap(scores: np.ndarray, targets: np.ndarray) -> float:
    positives = targets.sum()
    if positives == 0:
        return 0.0
    tp, fp = grouped_pr(scores, targets)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / positives
    prev_recall = np.r_[0.0, recall[:-1]]
    return float(((recall - prev_recall) * precision).sum())


def recall_at_fp_budget(scores: np.ndarray, targets: np.ndarray,
                        budget: int) -> float:
    positives = targets.sum()
    if positives == 0:
        return 0.0
    tp, fp = grouped_pr(scores, targets)
    within = fp <= budget
    if not within.any():
        return 0.0
    return float(tp[within][-1] / positives)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-sha", default=None,
                    help="sha256 of the checkpoint file, embedded for lineage")
    ap.add_argument("--config-sha", default=None,
                    help="canonical sha of the arm config, embedded for lineage")
    ap.add_argument("--protocol-sha", default=None,
                    help="combined sha over the evaluator AND its scoring "
                         "dependencies, computed by the caller; overrides the "
                         "single-file self-hash so the driver's cache key and "
                         "the embedded key can never disagree")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model = model.to(device).eval()

    loader = torch.utils.data.DataLoader(
        SpeakerCountDataset(args.manifest, feature_extractor=feature_extractor_for_config(config)),
        batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=4)

    all_scores, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            output, h_lens = model(batch["features"].to(device),
                                   batch["feature_lens"].to(device))
            boundary = getattr(output, "boundary_logits", None)
            if boundary is None:
                raise SystemExit(
                    "checkpoint's head exposes no boundary_logits — this "
                    "script is only meaningful for the pyramid head")
            aligned = align_batch_labels(batch["labels"].to(device),
                                         batch["label_lens"].to(device), h_lens)
            valid_frame, valid_pair = build_valid_masks(
                aligned.cpu(), h_lens.cpu(), expected_T=aligned.shape[1])
            labels = aligned.cpu()
            change = torch.zeros_like(valid_pair)
            change[:, 1:] = valid_pair[:, 1:] & (labels[:, 1:] != labels[:, :-1])
            scores = boundary.float().squeeze(-1).cpu()   # RAW logits
            if not torch.isfinite(scores[valid_pair]).all():
                raise SystemExit(
                    "non-finite boundary logits on valid transitions — the "
                    "checkpoint is unfit to gate on; refusing to score")
            all_scores.append(scores[valid_pair].numpy())
            all_targets.append(change[valid_pair].numpy().astype(np.int64))

    scores = np.concatenate(all_scores)
    targets = np.concatenate(all_targets)
    n_pos = int(targets.sum())
    result = {
        "boundary_ap": tie_aware_ap(scores, targets),
        "recall_at_fp_budget": recall_at_fp_budget(scores, targets, n_pos),
        "fp_budget": n_pos,
        "prevalence": float(targets.mean()),
        "n_transitions": n_pos,
        "n_scored": int(len(targets)),
        "checkpoint": args.checkpoint,
        "checkpoint_sha": args.checkpoint_sha,
        "config_sha": args.config_sha,
        # cache keys computed HERE so a cached report can never outlive a
        # manifest edit or a change to this evaluator's own scoring rules
        "manifest_sha": _sha_of_file(args.manifest),
        "protocol_sha": (args.protocol_sha if args.protocol_sha
                         else _sha_of_file(os.path.abspath(__file__))),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"boundary AP={result['boundary_ap']:.4f} "
          f"recall@FP<= {n_pos}={result['recall_at_fp_budget']:.4f} "
          f"(prevalence {result['prevalence']:.4f}) -> {args.out}")


if __name__ == "__main__":
    main()
