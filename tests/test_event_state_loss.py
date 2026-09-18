"""Regression tests for signed event targets and event-state auxiliary losses."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.losses import ZipCountLoss


def _loss(**kwargs):
    return ZipCountLoss(
        count_loss_type="ce",
        lambda_vad=0.0,
        lambda_overlap=0.0,
        lambda_smooth=0.0,
        lambda_event=kwargs.pop("lambda_event", 1.0),
        event_class_weights=kwargs.pop("event_class_weights", [1.0, 1.0, 1.0]),
        lambda_raw_anchor=kwargs.pop("lambda_raw_anchor", 0.0),
        **kwargs,
    )


def test_event_target_order_and_first_frame_mask():
    # Changes at t=1..4 are STAY, UP, STAY, DOWN.
    labels = torch.tensor([[0, 0, 1, 1, 0]])
    valid = torch.ones_like(labels, dtype=torch.bool)
    event = torch.full((1, 5, 3), -12.0)
    event[0, 0, 0] = 12.0  # deliberately wrong; crop frame zero is ignored
    event[0, 1, 1] = 12.0
    event[0, 2, 2] = 12.0
    event[0, 3, 1] = 12.0
    event[0, 4, 0] = 12.0
    got = _loss()._event_loss(event, labels, valid)
    assert got < 1e-6, got


def test_event_padding_and_no_pair_are_zero_safe():
    loss_fn = _loss()
    labels = torch.tensor([[1, 2, -100]])
    valid = labels >= 0
    event = torch.tensor([[[100.0, -100.0, -100.0],
                           [-100.0, -100.0, 100.0],
                           [100.0, -100.0, -100.0]]])
    got = loss_fn._event_loss(event, labels, valid)
    assert got < 1e-6, got  # only the valid 1->2 UP pair contributes

    singleton = torch.tensor([[2]])
    zero = loss_fn._event_loss(torch.randn(1, 1, 3), singleton,
                               torch.ones_like(singleton, dtype=torch.bool))
    assert zero.item() == 0.0


def test_extended_output_has_finite_gradients():
    torch.manual_seed(0)
    raw = torch.randn(2, 9, 4, requires_grad=True)
    filtered = torch.log_softmax(raw.float(), dim=-1)
    vad = torch.randn(2, 9, 1, requires_grad=True)
    overlap = torch.randn(2, 9, 1, requires_grad=True)
    event = torch.randn(2, 9, 3, requires_grad=True)
    labels = torch.tensor([
        [0, 0, 1, 1, 2, 2, 1, 1, 0],
        [1, 1, 1, 2, 2, 3, 3, -100, -100],
    ])
    h_lens = torch.tensor([9, 7])
    out = _loss(lambda_event=0.2, lambda_raw_anchor=0.25)(
        (filtered, vad, overlap, raw, event), labels, h_lens)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    for tensor in (raw, vad, overlap, event):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_event_config_rejects_legacy_head():
    logits = (torch.randn(1, 4, 4), torch.randn(1, 4, 1), torch.randn(1, 4, 1))
    try:
        _loss(lambda_event=0.2)(logits, torch.zeros(1, 4, dtype=torch.long),
                                torch.tensor([4]))
    except ValueError as exc:
        assert "event logits" in str(exc)
    else:
        raise AssertionError("event loss silently accepted a legacy 3-output head")


def test_legacy_three_output_head_still_works_by_default():
    logits = tuple(torch.randn(1, 4, d) for d in (4, 1, 1))
    out = _loss(lambda_event=0.0)(
        logits, torch.tensor([[0, 1, 2, 1]]), torch.tensor([4]))
    assert torch.isfinite(out["loss"])
