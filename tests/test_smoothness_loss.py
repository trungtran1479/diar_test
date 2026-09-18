"""Regression tests for ZipCountLoss._smoothness_loss's KL branch.

The old formulation called F.kl_div(log(p + 1e-8), q) directly. Its forward
value looks finite (the +1e-8 clamp keeps log(0) from literally appearing),
but its backward pass differentiates through log(target) where target can
be an exact 0.0 -- routine at fp16, reachable at fp32 with large-magnitude
logits -- producing a non-finite gradient even though the loss VALUE was
finite. Found via user review with logits in [-100, 100]; reproduced here
independently of any specific training run.

The fix rewrites the symmetric KL as the mathematically equivalent Jeffreys
divergence sum (p-q)(log p - log q), computed from a single numerically
stable log_softmax call, so no term ever requires log(0) of a raw
probability.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.losses import ZipCountLoss  # noqa: E402


def _loss_fn(smooth_type="kl"):
    fn = ZipCountLoss.__new__(ZipCountLoss)
    fn.smooth_type = smooth_type
    return fn


def test_extreme_logits_fp32_loss_and_grad_finite():
    torch.manual_seed(0)
    logits = (torch.rand(2, 8, 4) * 2 - 1) * 100
    logits.requires_grad_(True)
    mask = torch.ones(2, 8, dtype=torch.bool)

    loss = _loss_fn()._smoothness_loss(logits, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_extreme_logits_fp16_loss_and_grad_finite():
    torch.manual_seed(0)
    logits = ((torch.rand(2, 8, 4) * 2 - 1) * 100).half()
    logits.requires_grad_(True)
    mask = torch.ones(2, 8, dtype=torch.bool)

    loss = _loss_fn()._smoothness_loss(logits, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_padding_and_internal_gap_excluded_from_pairs():
    """A pair is only scored if BOTH frames are valid -- a pair straddling
    a gap (one valid frame next to one invalid frame) must not leak into
    the loss, matching the mask-intersection fix (mask[:,1:] & mask[:,:-1])
    rather than the old mask[:,1:]-only convention."""
    torch.manual_seed(1)
    logits = torch.randn(1, 6, 4, requires_grad=True)
    mask = torch.tensor([[True, True, False, True, True, True]])  # internal gap at t=2

    loss = _loss_fn()._smoothness_loss(logits, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()

    # Manually recompute over only the pairs that should survive: (0,1),(3,4),(4,5)
    import torch.nn.functional as F
    logp = F.log_softmax(logits.detach().float(), dim=-1)
    p = logp.exp()
    valid_pairs = [(0, 1), (3, 4), (4, 5)]
    terms = [
        ((p[0, b] - p[0, a]) * (logp[0, b] - logp[0, a])).sum()
        for a, b in valid_pairs
    ]
    expected = torch.stack(terms).mean()
    assert torch.allclose(loss, expected, atol=1e-5)


def test_matches_manual_jeffreys_on_mild_logits():
    torch.manual_seed(2)
    logits = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)

    got = _loss_fn()._smoothness_loss(logits, mask)

    import torch.nn.functional as F
    p1 = F.softmax(logits[:, 1:], dim=-1)
    p0 = F.softmax(logits[:, :-1], dim=-1)
    manual = (p1 * (p1.log() - p0.log())).sum(-1) + (p0 * (p0.log() - p1.log())).sum(-1)
    assert torch.allclose(got, manual.mean(), atol=1e-4)


def test_l1_branch_still_finite_and_unaffected():
    torch.manual_seed(3)
    logits = torch.randn(2, 5, 4, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.bool)
    loss = _loss_fn(smooth_type="l1")._smoothness_loss(logits, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_single_frame_returns_zero():
    fn = _loss_fn()
    logits = torch.randn(1, 1, 4)
    mask = torch.ones(1, 1, dtype=torch.bool)
    loss = fn._smoothness_loss(logits, mask)
    assert loss.item() == 0.0
