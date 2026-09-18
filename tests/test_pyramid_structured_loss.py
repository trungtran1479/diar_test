"""Focused invariants for the structured ordinal/segment objectives."""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.pyramid_head import GatedPyramidOrdinalHead  # noqa: E402
from src.models.structured_losses import (  # noqa: E402
    PyramidStructuredLoss,
    balanced_soft_bce_with_logits,
    build_valid_masks,
    corn_log_probs,
    soft_boundary_targets,
)


def _corn_outputs(
    ordinal: torch.Tensor,
    *,
    stage1: torch.Tensor | None = None,
    boundary: torch.Tensor | None = None,
    direction: torch.Tensor | None = None,
):
    batch, frames, _ = ordinal.shape
    if stage1 is None:
        stage1 = ordinal
    if boundary is None:
        boundary = ordinal.new_zeros(batch, frames, 1)
    if direction is None:
        direction = ordinal.new_zeros(batch, frames, 1)
    return {
        "ordinal_logits": ordinal,
        "stage1_ordinal_logits": stage1,
        "boundary_logits": boundary,
        "direction_logits": direction,
    }


def test_strict_frame_and_pair_masks_include_both_adjacent_frames():
    labels = torch.tensor(
        [
            [0, 1, -100, 2, 3],
            [1, 1, 2, 3, -100],
        ]
    )
    frame, pair = build_valid_masks(labels, torch.tensor([5, 3]), expected_T=5)
    torch.testing.assert_close(
        frame,
        torch.tensor(
            [
                [True, True, False, True, True],
                [True, True, True, False, False],
            ]
        ),
    )
    torch.testing.assert_close(
        pair,
        torch.tensor(
            [
                [False, True, False, False, True],
                [False, True, True, False, False],
            ]
        ),
    )

    with pytest.raises(ValueError, match="time mismatch"):
        build_valid_masks(labels, torch.tensor([5, 3]), expected_T=4)
    with pytest.raises(TypeError, match="integer dtype"):
        build_valid_masks(labels.float(), torch.tensor([5, 3]))
    with pytest.raises(ValueError, match=r"\[0,5\]"):
        build_valid_masks(labels, torch.tensor([6, 3]))
    bad = labels.clone()
    bad[0, 0] = 4
    with pytest.raises(ValueError, match=r"\[0,3\]"):
        build_valid_masks(bad, torch.tensor([5, 3]))


def test_soft_boundary_envelope_never_crosses_an_internal_gap():
    labels = torch.tensor([[0, 0, 1, -100, 1, 2, 2]])
    _, pair = build_valid_masks(labels, torch.tensor([7]))
    target = soft_boundary_targets(labels, pair)
    expected = torch.tensor([[0.0, 0.75, 1.0, 0.0, 0.0, 1.0, 0.75]])
    torch.testing.assert_close(target, expected, atol=0, rtol=0)
    assert target[0, 4] == 0  # no mass from either side traverses the gap


def test_loss_and_architecture_share_exact_corn_posterior():
    torch.manual_seed(0)
    head = GatedPyramidOrdinalHead(
        d_in=6,
        stack_dims=(1, 1, 1, 1, 1, 1),
        final_dim=None,
        d_model=8,
        dilations=(1, 2, 4, 8, 16, 32),
        dropout=0.0,
        ordinal_mode="corn",
        use_stage2=False,
    )
    logits = torch.randn(2, 17, 3) * 7.0
    _, architecture_logp = head.stage1_ordinal.decode(logits)
    loss_logp = corn_log_probs(logits)
    torch.testing.assert_close(
        architecture_logp, loss_logp, atol=0, rtol=0
    )
    probs = loss_logp.exp()
    q = torch.stack(
        [probs[..., 1:].sum(-1), probs[..., 2:].sum(-1), probs[..., 3]],
        dim=-1,
    )
    assert torch.all(q[..., 0] >= q[..., 1])
    assert torch.all(q[..., 1] >= q[..., 2])


def test_corn_normalizes_each_supported_conditional_risk_independently():
    objective = PyramidStructuredLoss(
        count_loss_type="corn", class_weights=torch.ones(4)
    )
    labels = torch.tensor([[0, 0, 0, 1, 2, 3]])
    valid = torch.ones_like(labels, dtype=torch.bool)
    logits = torch.zeros(1, 6, 3)
    logits[..., 0] = 0.0
    logits[..., 1] = -2.0
    logits[..., 2] = 3.0

    actual = objective._corn_loss(logits, labels, valid)
    risk = torch.stack(
        [valid, valid & (labels >= 1), valid & (labels >= 2)], dim=-1
    )
    targets = torch.stack(
        [labels >= 1, labels >= 2, labels >= 3], dim=-1
    ).float()
    raw = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    expected = torch.stack(
        [raw[..., k][risk[..., k]].mean() for k in range(3)]
    ).mean()
    pooled = raw[risk].mean()
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
    assert abs(float(actual - pooled)) > 0.05

    # If only k=1 has support it is the whole loss, not one third of it.
    only_zero = torch.zeros(1, 4, dtype=torch.long)
    only_valid = torch.ones_like(only_zero, dtype=torch.bool)
    only_logits = torch.full((1, 4, 3), 1.25)
    single = objective._corn_loss(only_logits, only_zero, only_valid)
    expected_single = F.binary_cross_entropy_with_logits(
        only_logits[..., 0], torch.zeros_like(only_logits[..., 0])
    )
    torch.testing.assert_close(single, expected_single, atol=1e-7, rtol=1e-7)


def test_cumulative_softmax_matches_three_manual_unconditional_risks():
    """Pure loss ablation: keep one four-logit softmax, only change geometry."""

    class_weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
    objective = PyramidStructuredLoss(
        count_loss_type="cumulative", class_weights=class_weights
    )
    logits = torch.tensor(
        [
            [
                [1.0, -0.5, 0.2, -1.0],
                [-1.0, 0.5, 1.5, 0.0],
                [0.1, 0.2, 0.3, 0.4],
                [2.0, -1.0, -2.0, -3.0],
                [-0.5, 0.1, 0.8, 1.7],
            ]
        ]
    )
    labels = torch.tensor([[0, 2, -100, 1, 3]])
    valid = labels >= 0
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels))
    logp = F.log_softmax(logits, dim=-1)

    actual = objective._cumulative_softmax_loss(logp, labels, valid)
    weights = torch.where(
        valid, class_weights[safe_labels], torch.zeros_like(logp[..., 0])
    )
    denominator = weights.sum()
    manual = []
    for threshold in (1, 2, 3):
        target = labels >= threshold
        positive = -torch.logsumexp(logp[..., threshold:], dim=-1)
        negative = -torch.logsumexp(logp[..., :threshold], dim=-1)
        per_frame = torch.where(target, positive, negative)
        per_frame = torch.where(valid, per_frame, torch.zeros_like(per_frame))
        manual.append((per_frame * weights).sum() / denominator)
    expected = torch.stack(manual).mean()
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)


def test_all_invalid_batch_is_exact_zero_with_finite_zero_gradients():
    tensors = [
        torch.full((2, 5, 3), float("nan"), requires_grad=True),
        torch.full((2, 5, 3), float("nan"), requires_grad=True),
        torch.full((2, 5, 1), float("nan"), requires_grad=True),
        torch.full((2, 5, 1), float("nan"), requires_grad=True),
    ]
    outputs = _corn_outputs(
        tensors[0], stage1=tensors[1], boundary=tensors[2], direction=tensors[3]
    )
    objective = PyramidStructuredLoss(
        count_loss_type="corn",
        lambda_stage1=0.3,
        lambda_boundary=0.1,
        lambda_direction=0.1,
        lambda_t_mse=0.1,
        lambda_segment=0.1,
        lambda_delta=0.1,
    )
    result = objective(
        outputs,
        torch.full((2, 5), -100, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
    )
    for value in result.values():
        assert torch.isfinite(value) and float(value) == 0.0
    result["loss"].backward()
    for tensor in tensors:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) == 0


def test_t1_has_main_loss_but_no_transition_or_segment_penalty():
    ordinal = torch.zeros(2, 1, 3, requires_grad=True)
    boundary = torch.zeros(2, 1, 1, requires_grad=True)
    direction = torch.zeros(2, 1, 1, requires_grad=True)
    objective = PyramidStructuredLoss(
        count_loss_type="corn",
        lambda_boundary=1.0,
        lambda_direction=1.0,
        lambda_t_mse=1.0,
        lambda_segment=1.0,
        lambda_delta=1.0,
    )
    result = objective(
        _corn_outputs(ordinal, boundary=boundary, direction=direction),
        torch.tensor([[0], [3]]),
        torch.tensor([1, 1]),
    )
    assert result["loss_count"] > 0
    for key in (
        "loss_boundary",
        "loss_direction",
        "loss_t_mse",
        "loss_segment",
        "loss_delta",
    ):
        assert float(result[key]) == 0.0
    result["loss"].backward()
    assert ordinal.grad is not None and torch.isfinite(ordinal.grad).all()


def test_temporal_mse_ignores_boundaries_padding_and_nan_gap():
    objective = PyramidStructuredLoss(
        count_loss_type="corn", t_mse_tau=10.0
    )
    logp = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [100.0, 100.0, 100.0, 100.0],
                [float("nan")] * 4,
                [2.0, 2.0, 2.0, 2.0],
                [4.0, 4.0, 4.0, 4.0],
            ]
        ],
        requires_grad=True,
    )
    pair = torch.tensor([[False, True, True, False, False, True]])
    soft_boundary = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    actual = objective._temporal_mse(logp, soft_boundary, pair)
    # Valid stable pairs: 0->1 has squared difference 1; 4->5 has 4.
    torch.testing.assert_close(actual, torch.tensor(2.5), atol=0, rtol=0)
    actual.backward()
    assert torch.isfinite(logp.grad).all()
    assert torch.count_nonzero(logp.grad[:, 2:4]) == 0


def test_segment_consistency_weights_short_and_long_segments_equally():
    probs = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        ]
    )
    labels = torch.tensor([[0, 0, 1, 1, 1, 1]])
    valid = torch.ones_like(labels, dtype=torch.bool)
    actual = PyramidStructuredLoss._segment_consistency(probs, labels, valid)
    # Segment 0 variance is .125; segment 1 variance is zero. Equal segments
    # therefore give .0625 (a frame-weighted implementation gives .041666...).
    torch.testing.assert_close(actual, torch.tensor(0.0625), atol=0, rtol=0)


def test_direction_is_supervised_only_at_true_change_centers():
    objective = PyramidStructuredLoss(count_loss_type="corn")
    labels = torch.tensor([[1, 1, 2, 2, 1]])
    pair = torch.tensor([[False, True, True, True, True]])
    logits = torch.zeros(1, 5, requires_grad=True)
    base = objective._direction_loss(logits, labels, pair)

    modified = logits.detach().clone()
    modified[0, 0] = 100.0
    modified[0, 1] = -100.0
    modified[0, 3] = 100.0
    changed = objective._direction_loss(modified, labels, pair)
    torch.testing.assert_close(base, changed, atol=0, rtol=0)

    base.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[0, [0, 1, 3]]) == 0
    assert torch.count_nonzero(logits.grad[0, [2, 4]]) == 2
    torch.testing.assert_close(base, torch.tensor(math.log(2.0)))


def test_delta_consistency_has_symmetric_half_detached_gradients():
    objective = PyramidStructuredLoss(count_loss_type="corn")
    # E[count] moves 1.0 -> 1.4, while event expectation is .25:
    # sigmoid(boundary)=.5 and signed direction=2*.75-1=.5.
    probs = torch.tensor(
        [[[0.0, 1.0, 0.0, 0.0], [0.0, 0.6, 0.4, 0.0]]],
        requires_grad=True,
    )
    boundary = torch.tensor([[0.0, 0.0]], requires_grad=True)
    direction = torch.tensor([[0.0, math.log(3.0)]], requires_grad=True)
    pair = torch.tensor([[False, True]])
    loss = objective._delta_consistency(probs, boundary, direction, pair)
    loss.backward()

    # Difference .15 is in Huber's quadratic region. Symmetric detach assigns
    # half its gradient to each side: dL/d(count_delta)=.075 and
    # dL/d(event_delta)=-.075.
    classes = torch.arange(4, dtype=torch.float32)
    torch.testing.assert_close(
        probs.grad[0, 1], 0.075 * classes, atol=2e-7, rtol=1e-6
    )
    torch.testing.assert_close(
        probs.grad[0, 0], -0.075 * classes, atol=2e-7, rtol=1e-6
    )
    torch.testing.assert_close(
        boundary.grad[0, 1], torch.tensor(-0.009375), atol=2e-7, rtol=1e-6
    )
    torch.testing.assert_close(
        direction.grad[0, 1],
        torch.tensor(-0.0140625),
        atol=2e-7,
        rtol=1e-6,
    )
    assert boundary.grad[0, 0] == 0 and direction.grad[0, 0] == 0


def test_all_auxiliary_lambdas_zero_is_main_objective_identity():
    ordinal = torch.randn(2, 7, 3, requires_grad=True)
    stage1 = torch.full((2, 7, 3), float("nan"), requires_grad=True)
    boundary = torch.full((2, 7, 1), float("nan"), requires_grad=True)
    direction = torch.full((2, 7, 1), float("nan"), requires_grad=True)
    objective = PyramidStructuredLoss(count_loss_type="corn")
    result = objective(
        _corn_outputs(
            ordinal,
            stage1=stage1,
            boundary=boundary,
            direction=direction,
        ),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3], [1, 1, 0, 0, -100, -100, -100]]),
        torch.tensor([7, 4]),
    )
    torch.testing.assert_close(result["loss"], result["loss_count"], atol=0, rtol=0)
    for key in (
        "loss_stage1",
        "loss_boundary",
        "loss_direction",
        "loss_t_mse",
        "loss_segment",
        "loss_delta",
    ):
        assert float(result[key]) == 0.0
    result["loss"].backward()
    assert ordinal.grad is not None and torch.isfinite(ordinal.grad).all()
    assert stage1.grad is None and boundary.grad is None and direction.grad is None


def test_balanced_boundary_bce_handles_no_positive_and_no_valid_frames():
    logits = torch.tensor([[2.0, -1.0, 0.5]], requires_grad=True)
    targets = torch.zeros_like(logits)
    valid = torch.tensor([[True, True, True]])
    loss = balanced_soft_bce_with_logits(logits, targets, valid)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()

    invalid_logits = torch.full((1, 3), float("nan"), requires_grad=True)
    zero = balanced_soft_bce_with_logits(
        invalid_logits, targets, torch.zeros_like(valid)
    )
    assert torch.isfinite(zero) and float(zero) == 0.0
    zero.backward()
    assert torch.isfinite(invalid_logits.grad).all()


def test_fp16_mixed_validity_full_loss_and_gradients_are_finite():
    torch.manual_seed(4)
    ordinal = torch.randn(2, 8, 3, dtype=torch.float16, requires_grad=True)
    stage1 = torch.randn(2, 8, 3, dtype=torch.float16, requires_grad=True)
    boundary = torch.randn(2, 8, 1, dtype=torch.float16, requires_grad=True)
    direction = torch.randn(2, 8, 1, dtype=torch.float16, requires_grad=True)
    labels = torch.tensor(
        [
            [0, 1, -100, 2, 2, 3, -100, -100],
            [1, 1, 2, 2, 1, 0, 0, 0],
        ]
    )
    objective = PyramidStructuredLoss(
        count_loss_type="corn",
        lambda_stage1=0.3,
        lambda_boundary=0.1,
        lambda_direction=0.1,
        lambda_t_mse=0.05,
        lambda_segment=0.05,
        lambda_delta=0.02,
    )
    result = objective(
        _corn_outputs(
            ordinal,
            stage1=stage1,
            boundary=boundary,
            direction=direction,
        ),
        labels,
        torch.tensor([6, 8]),
    )
    assert all(torch.isfinite(value) for value in result.values())
    result["loss"].backward()
    for tensor in (ordinal, stage1, boundary, direction):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
