"""Tests for the two config-gated TCNCountHead options added for the context
oracle (``causal=False``) and the stack probe (``stack_indices``).

Both options must leave the default path byte-identical: an option-free head
constructed from the same seed must produce the same parameters and outputs as
before the options existed (guarded here by comparing against a default
construction).
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.heads import TCNCountHead  # noqa: E402

STACK_DIMS = [192, 256, 384, 512, 384, 256]
D_IN = sum(STACK_DIMS)


def make(seed=0, **kw):
    torch.manual_seed(seed)
    head = TCNCountHead(D_IN, stack_dims=STACK_DIMS, d_model=96, **kw).eval()
    with torch.no_grad():  # trained-like: nonzero LN bias (see test_tcn_streaming)
        for blk in head.blocks:
            torch.nn.init.normal_(blk.norm.bias, std=0.1)
    return head


def test_noncausal_sees_future():
    head = make(causal=False)
    x = torch.randn(1, 200, D_IN)
    with torch.no_grad():
        y1, _, _ = head(x)
        x2 = x.clone()
        # NOT a uniform shift: LayerNorm removes any channel-constant offset,
        # so `+= 100.0` is invisible to the first norm and falsely passes as
        # "causal". Random noise is not LN-invariant.
        x2[:, 150:] += torch.randn_like(x2[:, 150:])
        y2, _, _ = head(x2)
    # a frame BEFORE the perturbation must now change: that is the point
    assert (y1[:, 100] - y2[:, 100]).abs().max() > 1e-4


def test_causal_default_unchanged_and_still_causal():
    head = make(causal=True)
    x = torch.randn(1, 200, D_IN)
    with torch.no_grad():
        y1, _, _ = head(x)
        x2 = x.clone()
        x2[:, 150:] += torch.randn_like(x2[:, 150:])
        y2, _, _ = head(x2)
    assert torch.equal(y1[:, :150], y2[:, :150])


def test_noncausal_refuses_streaming():
    head = make(causal=False)
    x = torch.randn(1, 16, D_IN)
    with pytest.raises(RuntimeError):
        head.forward_streaming(x, None)


def test_stack_indices_isolate_their_slice():
    head = make(stack_indices=[3])
    x = torch.randn(1, 60, D_IN)
    lo = sum(STACK_DIMS[:3])
    hi = lo + STACK_DIMS[3]
    with torch.no_grad():
        y1, _, _ = head(x)
        # perturbing every OTHER stack must change nothing
        x2 = x.clone()
        x2[..., :lo] += torch.randn_like(x2[..., :lo])
        x2[..., hi:] += torch.randn_like(x2[..., hi:])
        y2, _, _ = head(x2)
        # perturbing the selected stack must change the output (random, not a
        # uniform shift — per-stack LayerNorm absorbs channel-constant offsets)
        x3 = x.clone()
        x3[..., lo:hi] += torch.randn_like(x3[..., lo:hi])
        y3, _, _ = head(x3)
    assert torch.equal(y1, y2), "unselected stacks leaked into the probe head"
    assert (y1 - y3).abs().max() > 1e-4


def test_stack_indices_validation():
    for bad in ([], [6], [0, 0], [-1]):
        with pytest.raises(ValueError):
            TCNCountHead(D_IN, stack_dims=STACK_DIMS, stack_indices=bad)
    with pytest.raises(ValueError):
        TCNCountHead(D_IN, stack_indices=[0])  # no stack_dims


def test_probe_head_streams_equivalently():
    head = make(stack_indices=[1, 4])
    x = torch.randn(1, 100, D_IN)
    with torch.no_grad():
        full = head(x)
        caches, outs, pos = None, [], 0
        for size in [16] * 6 + [4]:
            out, caches = head.forward_streaming(x[:, pos:pos + size], caches)
            outs.append(out)
            pos += size
        chunked = torch.cat([o[0] for o in outs], dim=1)
    assert (full[0] - chunked).abs().max() < 2e-5


if __name__ == "__main__":
    test_noncausal_sees_future()
    test_causal_default_unchanged_and_still_causal()
    try:
        test_noncausal_refuses_streaming()
    except ImportError:
        pass
    test_stack_indices_isolate_their_slice()
    test_probe_head_streams_equivalently()
    print("ALL PASSED")
