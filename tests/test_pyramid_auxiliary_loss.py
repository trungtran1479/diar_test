"""PyramidAuxiliaryLoss: boundary/direction/t_mse/segment ONLY, no count term.

This is the legacy-bridge alternative to PyramidStructuredLoss's SORD/CORN
switch (which failed its own non-inferiority bar in the Phase-8 chain,
phase b). The invariant this file protects is numerical: composing this
loss with the unmodified legacy ZipCountLoss must reproduce EXACTLY the
same boundary/direction/t_mse/segment values PyramidStructuredLoss already
computes and has tested, since both delegate to the same free functions.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.structured_losses import (  # noqa: E402
    PyramidAuxiliaryLoss,
    PyramidStructuredLoss,
)


def _outputs(ordinal, boundary=None, direction=None):
    batch, frames, _ = ordinal.shape
    log_probs = torch.log_softmax(ordinal, dim=-1)
    if boundary is None:
        boundary = ordinal.new_zeros(batch, frames, 1)
    if direction is None:
        direction = ordinal.new_zeros(batch, frames, 1)
    return {
        "count_logits": ordinal,
        "count_log_probs": log_probs,
        "boundary_logits": boundary,
        "direction_logits": direction,
    }


def _random_batch(seed=0, batch=2, frames=17):
    g = torch.Generator().manual_seed(seed)
    ordinal = torch.randn(batch, frames, 4, generator=g)
    boundary = torch.randn(batch, frames, 1, generator=g)
    direction = torch.randn(batch, frames, 1, generator=g)
    labels = torch.randint(0, 4, (batch, frames), generator=g)
    h_lens = torch.tensor([frames, frames - 3])
    return ordinal, boundary, direction, labels, h_lens


def test_zero_lambdas_is_a_differentiable_zero_without_optional_fields():
    aux = PyramidAuxiliaryLoss()
    ordinal, _, _, labels, h_lens = _random_batch()
    out = {"count_logits": ordinal, "count_log_probs": torch.log_softmax(ordinal, -1)}
    result = aux(out, labels, h_lens)
    assert float(result["loss"]) == 0.0
    assert result["loss"].requires_grad or ordinal.requires_grad is False
    for key in ("loss_boundary", "loss_direction", "loss_t_mse", "loss_segment"):
        assert float(result[key]) == 0.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_pyramid_structured_loss_component_values(seed):
    ordinal, boundary, direction, labels, h_lens = _random_batch(seed=seed)
    out = _outputs(ordinal, boundary, direction)

    aux = PyramidAuxiliaryLoss(
        lambda_boundary=0.1,
        lambda_direction=0.05,
        lambda_t_mse=0.05,
        lambda_segment=0.05,
        direction_class_weights=[1.0, 1.5],
        t_mse_tau=3.0,
    )
    reference = PyramidStructuredLoss(
        count_loss_type="ce",
        lambda_boundary=0.1,
        lambda_direction=0.05,
        lambda_t_mse=0.05,
        lambda_segment=0.05,
        direction_class_weights=[1.0, 1.5],
        t_mse_tau=3.0,
    )

    got = aux(out, labels, h_lens)
    want = reference(out, labels, h_lens)

    for key in ("loss_boundary", "loss_direction", "loss_t_mse", "loss_segment"):
        torch.testing.assert_close(got[key], want[key])
    expected_total = (
        0.1 * want["loss_boundary"]
        + 0.05 * want["loss_direction"]
        + 0.05 * want["loss_t_mse"]
        + 0.05 * want["loss_segment"]
    )
    torch.testing.assert_close(got["loss"], expected_total)
    # The auxiliary loss must never depend on the count objective at all.
    assert "loss_count" not in got


def test_missing_boundary_logits_raises_when_lambda_positive():
    aux = PyramidAuxiliaryLoss(lambda_boundary=0.1)
    ordinal, _, _, labels, h_lens = _random_batch()
    out = {"count_logits": ordinal, "count_log_probs": torch.log_softmax(ordinal, -1)}
    with pytest.raises(ValueError, match="boundary_logits"):
        aux(out, labels, h_lens)


def test_missing_direction_logits_raises_when_lambda_positive():
    aux = PyramidAuxiliaryLoss(lambda_direction=0.1)
    ordinal, _, _, labels, h_lens = _random_batch()
    out = {"count_logits": ordinal, "count_log_probs": torch.log_softmax(ordinal, -1)}
    with pytest.raises(ValueError, match="direction_logits"):
        aux(out, labels, h_lens)


def test_backward_reaches_boundary_and_direction_heads():
    ordinal, boundary, direction, labels, h_lens = _random_batch()
    boundary = boundary.requires_grad_(True)
    direction = direction.requires_grad_(True)
    out = _outputs(ordinal, boundary, direction)
    aux = PyramidAuxiliaryLoss(
        lambda_boundary=0.1, lambda_direction=0.05,
        lambda_t_mse=0.05, lambda_segment=0.05,
    )
    result = aux(out, labels, h_lens)
    result["loss"].backward()
    assert boundary.grad is not None and torch.count_nonzero(boundary.grad) > 0
    assert direction.grad is not None and torch.count_nonzero(direction.grad) > 0


def test_only_segment_lambda_needs_neither_boundary_nor_direction():
    aux = PyramidAuxiliaryLoss(lambda_segment=0.2)
    ordinal, _, _, labels, h_lens = _random_batch()
    out = {"count_logits": ordinal, "count_log_probs": torch.log_softmax(ordinal, -1)}
    result = aux(out, labels, h_lens)  # must not raise
    assert float(result["loss_segment"]) >= 0.0
    torch.testing.assert_close(result["loss"], 0.2 * result["loss_segment"])


def test_invalid_direction_class_weights_rejected():
    with pytest.raises(ValueError):
        PyramidAuxiliaryLoss(direction_class_weights=[1.0, -1.0])


def test_invalid_boundary_kernel_rejected():
    with pytest.raises(ValueError):
        PyramidAuxiliaryLoss(boundary_kernel=[0.5, 1.0])  # even length
