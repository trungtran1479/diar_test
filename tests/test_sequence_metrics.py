"""Segmental edit score, fragmentation rate, transition confusion, ECE/Brier.

chatgpt_recommend.txt section 11 flags these as the metrics macro-F1 blindly
misses (over-segmentation, boundary quality, confidence calibration). Hand-
worked examples anchor each function against its textbook definition.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.sequence_metrics import (  # noqa: E402
    aggregate_sequence_metrics,
    brier_score,
    count_mae,
    expected_calibration_error,
    fragmentation_rate,
    segmental_edit_score,
    to_segments,
    transition_confusion,
)


def test_to_segments_collapses_runs():
    assert to_segments([1, 1, 1, 2, 2, 1, 1]) == [1, 2, 1]
    assert to_segments([]) == []
    assert to_segments([0]) == [0]


def test_segmental_edit_score_perfect_match_is_100():
    seq = [0, 0, 1, 1, 1, 2, 2, 0]
    assert segmental_edit_score(seq, seq) == 100.0


def test_segmental_edit_score_matches_hand_worked_example():
    # true segments: [1,2,1]; pred segments: [1,2,3,1] -> one insertion
    true_seq = [1, 1, 2, 2, 1]
    pred_seq = [1, 1, 2, 2, 3, 1]
    # levenshtein([1,2,1],[1,2,3,1]) = 1 (insert 3); denom = max(3,4) = 4
    expected = (1.0 - 1 / 4) * 100.0
    assert segmental_edit_score(true_seq, pred_seq) == pytest.approx(expected)


def test_segmental_edit_score_is_duration_blind():
    # boundary shifted by many frames, but same segment ORDER -> still 100
    true_seq = [0] * 10 + [1] * 10
    pred_seq = [0] * 3 + [1] * 17
    assert segmental_edit_score(true_seq, pred_seq) == 100.0


def test_segmental_edit_score_empty_sequences_is_100():
    assert segmental_edit_score([], []) == 100.0


def test_fragmentation_rate_flickering_prediction_is_greater_than_one():
    true_seq = [1] * 20
    pred_seq = [1, 2] * 10  # 20 alternating segments vs 1 true segment
    assert fragmentation_rate(true_seq, pred_seq) == pytest.approx(20.0)


def test_fragmentation_rate_oversmoothed_prediction_is_less_than_one():
    true_seq = [0, 1, 0, 1, 0]  # 5 segments
    pred_seq = [0, 0, 0, 0, 0]  # 1 segment
    assert fragmentation_rate(true_seq, pred_seq) == pytest.approx(0.2)


def test_fragmentation_rate_ideal_is_one():
    true_seq = [0, 0, 1, 1, 2]
    pred_seq = [0, 0, 0, 1, 2]  # same 3 segments, different boundary
    assert fragmentation_rate(true_seq, pred_seq) == pytest.approx(1.0)


def test_transition_confusion_perfect_prediction_is_diagonal():
    seq = [0, 1, 1, 2, 1, 0]  # deltas: +1, 0, +1, -1, -1
    result = transition_confusion(seq, seq)
    matrix = np.asarray(result["matrix"])
    assert result["labels"] == ["down", "stable", "up"]
    # off-diagonal must be exactly zero for a self-comparison
    assert np.array_equal(matrix, np.diag(np.diag(matrix)))
    assert matrix.sum() == len(seq) - 1


def test_transition_confusion_magnitude_agnostic_direction():
    true_seq = [3, 1]   # a -2 drop
    pred_seq = [1, 0]   # a -1 drop: same DOWN bucket despite different magnitude
    result = transition_confusion(true_seq, pred_seq)
    matrix = np.asarray(result["matrix"])
    assert matrix[0, 0] == 1  # true=down, pred=down
    assert matrix.sum() == 1


def test_transition_confusion_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        transition_confusion([0, 1, 2], [0, 1])


def test_ece_is_zero_for_perfectly_calibrated_predictions():
    # confidence always matches empirical accuracy exactly (degenerate: every
    # prediction is 100% confident and always correct)
    probs = np.array([[0.0, 1.0, 0.0, 0.0]] * 10)
    labels = np.array([1] * 10)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)


def test_ece_penalizes_overconfidence():
    # 100% confident but only right half the time -> |acc - conf| = 0.5
    probs = np.array([[0.0, 1.0, 0.0, 0.0]] * 10)
    labels = np.array([1, 0] * 5)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.5)


def test_ece_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        expected_calibration_error(np.zeros((5, 4)), np.zeros(3))


def test_count_mae_matches_hand_worked_example():
    # argmax predictions: 1, 0, 3 vs labels 0, 0, 1 -> |1|+|0|+|2| = 3 / 3 = 1.0
    probs = np.array([
        [0.1, 0.9, 0.0, 0.0],
        [0.7, 0.3, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    labels = np.array([0, 0, 1])
    assert count_mae(probs, labels) == pytest.approx(1.0)


def test_count_mae_zero_for_perfect_predictions():
    probs = np.array([[0, 1, 0, 0], [0, 0, 0, 1]])
    labels = np.array([1, 3])
    assert count_mae(probs, labels) == pytest.approx(0.0)


def test_count_mae_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        count_mae(np.zeros((5, 4)), np.zeros(3))


def test_brier_score_zero_for_perfect_onehot_prediction():
    probs = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    labels = np.array([1, 0])
    assert brier_score(probs, labels) == pytest.approx(0.0)


def test_brier_score_matches_hand_worked_example():
    probs = np.array([[0.25, 0.25, 0.25, 0.25]])
    labels = np.array([0])
    # (0.75^2 + 0.25^2*3) = 0.5625 + 0.1875 = 0.75
    assert brier_score(probs, labels) == pytest.approx(0.75)


def test_aggregate_sequence_metrics_pools_across_recordings():
    per_rec = {
        "a": {
            "edit_score": 100.0,
            "fragmentation_rate": 1.0,
            "transition_confusion": {
                "labels": ["down", "stable", "up"],
                "matrix": [[1, 0, 0], [0, 2, 0], [0, 0, 1]],
            },
        },
        "b": {
            "edit_score": 50.0,
            "fragmentation_rate": float("nan"),
            "transition_confusion": {
                "labels": ["down", "stable", "up"],
                "matrix": [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            },
        },
    }
    out = aggregate_sequence_metrics(per_rec)
    assert out["mean_edit_score"] == pytest.approx(75.0)
    assert out["mean_fragmentation_rate"] == pytest.approx(1.0)  # NaN dropped
    assert out["pooled_transition_confusion"]["matrix"] == [
        [1, 1, 0], [0, 2, 0], [0, 0, 1],
    ]
