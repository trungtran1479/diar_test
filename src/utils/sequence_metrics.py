"""Sequence-level metrics that frame-flattened confusion counts cannot see.

``metrics.py`` treats every frame as an i.i.d. sample, which is exactly right
for F1/MAE/calibration but structurally blind to over-segmentation: two
predictions can have near-identical frame accuracy while one flickers between
classes every few frames and the other tracks the true segment structure
(see PYRAMID_MASK_TRANSFER_PREREG-era discussion and chatgpt_recommend.txt
section 11). These functions require the FULL, temporally-ordered per-frame
label sequence of a recording -- callers are responsible for reconstructing
that order (see ``scripts/eval_sequence_metrics.py``, which stitches
consecutive fixed-length windows back into one sequence per recording).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def to_segments(label_seq: Sequence[int]) -> List[int]:
    """Collapse consecutive equal labels into one entry per run."""
    segments: List[int] = []
    for label in label_seq:
        if not segments or segments[-1] != label:
            segments.append(int(label))
    return segments


def _levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    """Standard edit distance (insert/delete/substitute, unit cost)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # delete from a
                curr[j - 1] + 1,    # insert into a
                prev[j - 1] + cost,  # substitute
            )
        prev = curr
    return prev[m]


def segmental_edit_score(
    true_seq: Sequence[int], pred_seq: Sequence[int]
) -> float:
    """MS-TCN-style edit score in [0, 100]; 100 = identical segment order.

    Computed on the collapsed segment-LABEL sequences (duration-blind), so
    a correct segment predicted with a slightly wrong boundary costs nothing
    here -- that is exactly the complement of the frame-level metrics.
    """
    true_segs = to_segments(true_seq)
    pred_segs = to_segments(pred_seq)
    denom = max(len(true_segs), len(pred_segs))
    if denom == 0:
        return 100.0
    distance = _levenshtein(true_segs, pred_segs)
    return (1.0 - distance / denom) * 100.0


def fragmentation_rate(
    true_seq: Sequence[int], pred_seq: Sequence[int]
) -> float:
    """#predicted segments / #ground-truth segments; 1.0 is ideal.

    >1 means the model is flickering (splitting true segments); <1 means it
    is over-smoothing (merging distinct segments together).
    """
    n_true = len(to_segments(true_seq))
    n_pred = len(to_segments(pred_seq))
    if n_true == 0:
        return float("nan")
    return n_pred / n_true


_DELTA_BUCKETS = ("down", "stable", "up")
_BUCKET_INDEX = {-1: 0, 0: 1, 1: 2}


def _delta_buckets(seq: Sequence[int]) -> np.ndarray:
    arr = np.asarray(seq, dtype=np.int64)
    if arr.size < 2:
        return np.zeros(0, dtype=np.int64)
    # sign() bucket by DIRECTION only: a 3->1 drop (two speakers stopping at
    # once) still counts as one DOWN event, matching chatgpt_recommend.txt
    # section 2.5's DOWN/STABLE/UP simplification of the count-change event.
    return np.sign(np.diff(arr)).astype(np.int64)


def transition_confusion(
    true_seq: Sequence[int], pred_seq: Sequence[int]
) -> Dict[str, object]:
    """3x3 DOWN/STABLE/UP confusion matrix over per-frame count-change events.

    Row = true bucket, column = predicted bucket, at matching frame
    transitions (t-1 -> t) in both sequences.
    """
    if len(true_seq) != len(pred_seq):
        raise ValueError(
            f"true_seq and pred_seq must be the same length, got "
            f"{len(true_seq)} vs {len(pred_seq)}"
        )
    true_b = _delta_buckets(true_seq)
    pred_b = _delta_buckets(pred_seq)
    matrix = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(true_b, pred_b):
        matrix[_BUCKET_INDEX[int(t)], _BUCKET_INDEX[int(p)]] += 1
    return {"labels": list(_DELTA_BUCKETS), "matrix": matrix.tolist()}


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Standard equal-width-bin ECE on the top-1 (argmax) confidence."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or probs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"probs must be [N,C] matching labels [N]; got {probs.shape} "
            f"vs {labels.shape}"
        )
    n = labels.shape[0]
    if n == 0:
        return float("nan")
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if lo == 0.0:
            mask = mask | (confidences == 0.0)
        if not np.any(mask):
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def count_mae(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean absolute error of the argmax count prediction (ordinal, not one-hot)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or probs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"probs must be [N,C] matching labels [N]; got {probs.shape} "
            f"vs {labels.shape}"
        )
    if labels.shape[0] == 0:
        return float("nan")
    predictions = probs.argmax(axis=1)
    return float(np.mean(np.abs(predictions - labels)))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and the one-hot label."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or probs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"probs must be [N,C] matching labels [N]; got {probs.shape} "
            f"vs {labels.shape}"
        )
    n_classes = probs.shape[1]
    onehot = np.eye(n_classes)[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def aggregate_sequence_metrics(
    per_recording: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    """Pool per-recording edit-score/fragmentation into corpus-level summary."""
    edit_scores = [row["edit_score"] for row in per_recording.values()]
    frag_rates = [
        row["fragmentation_rate"] for row in per_recording.values()
        if row["fragmentation_rate"] == row["fragmentation_rate"]  # drop NaN
    ]
    total_cm = np.zeros((3, 3), dtype=np.int64)
    for row in per_recording.values():
        total_cm += np.asarray(row["transition_confusion"]["matrix"])
    return {
        "mean_edit_score": float(np.mean(edit_scores)) if edit_scores else float("nan"),
        "mean_fragmentation_rate": (
            float(np.mean(frag_rates)) if frag_rates else float("nan")
        ),
        "pooled_transition_confusion": {
            "labels": list(_DELTA_BUCKETS),
            "matrix": total_cm.tolist(),
        },
    }
