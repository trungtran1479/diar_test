"""Core tests for the causal TCN event-state count head.

Run:
    PYTHONPATH=. python tests/test_event_state_streaming.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.heads import (  # noqa: E402
    CausalSignedStickyFilter,
    EventStateCache,
    TCNEventStateCountHead,
)
from src.models.zipcount_v1 import build_model  # noqa: E402


def build_head(seed=0):
    torch.manual_seed(seed)
    head = TCNEventStateCountHead(
        d_in=12,
        num_classes=4,
        d_model=8,
        dilations=(1, 2, 4),
        kernel=3,
        dropout=0.0,
        event_stay_bias=1.5,
    ).eval()
    # Load-bearing: streaming TCN cache must remain correct after LayerNorm
    # biases move away from their zero initialization during training.
    with torch.no_grad():
        for block in head.blocks:
            torch.nn.init.normal_(block.norm.bias, std=0.1)
    return head


def test_event_projection_starts_from_feature_independent_sticky_prior():
    head = build_head(seed=9)
    assert torch.count_nonzero(head.event_proj.weight) == 0
    torch.testing.assert_close(
        head.event_proj.bias, torch.tensor([0.0, 1.5, 0.0]))


def max_output_error(a, b):
    return max((x - y).abs().max().item() for x, y in zip(a, b))


def chunked(head, x, sizes):
    cache = None
    pieces = [[] for _ in range(5)]
    pos = 0
    with torch.no_grad():
        for size in sizes:
            out, cache = head.forward_streaming(x[:, pos:pos + size], cache)
            for i, value in enumerate(out):
                pieces[i].append(value)
            pos += size
    assert pos == x.shape[1], (pos, x.shape[1])
    return tuple(torch.cat(part, dim=1) for part in pieces), cache


def test_transition_invariants():
    filt = CausalSignedStickyFilter(num_classes=4)
    stay = torch.tensor([[-30.0, 30.0, -30.0]])
    up = torch.tensor([[-30.0, -30.0, 30.0]])
    down = torch.tensor([[30.0, -30.0, -30.0]])
    a_stay = filt.transition_matrix(stay)[0]
    a_up = filt.transition_matrix(up)[0]
    a_down = filt.transition_matrix(down)[0]

    for a in (a_stay, a_up, a_down):
        torch.testing.assert_close(a.sum(-1), torch.ones(4), atol=1e-6, rtol=0)
        assert (a >= 0).all()
    assert torch.diagonal(a_stay).min() > 0.999
    for i in range(3):
        assert a_up[i, :i + 1].max() < 1e-4
        assert a_up[i, i + 1] > 0.999
        if i + 2 < 4:
            assert a_up[i, i + 2:].max() < 1e-4
    assert a_up[3, 3] > 0.999  # impossible UP at ceiling folds into STAY
    for i in range(1, 4):
        assert a_down[i, i:].max() < 1e-4
        assert a_down[i, i - 1] > 0.999
        if i - 1 > 0:
            assert a_down[i, :i - 1].max() < 1e-4
    assert a_down[0, 0] > 0.999  # impossible DOWN at floor folds into STAY
    print("  transition rows/directions     OK")


def test_weighted_ce_logits_are_calibrated_before_transition():
    """If CE weights rare events by 12x, the filter must divide that shift out."""
    weights = torch.tensor([12.0, 1.0, 12.0])
    true_p = torch.tensor([0.005, 0.990, 0.005])
    # At the weighted-CE optimum q is proportional to weight * true_p.
    weighted_optimum_logits = (weights * true_p).log().unsqueeze(0)
    filt = CausalSignedStickyFilter(
        num_classes=4, event_class_weights=weights.tolist())
    a = filt.transition_matrix(weighted_optimum_logits)[0]
    # Interior state 1 has direct DOWN/STAY/UP destinations 0/1/2.
    torch.testing.assert_close(a[1, :3], true_p, atol=5e-6, rtol=0)
    assert a[1, 3] < 2e-6
    print("  weighted-CE calibration       OK")


def test_known_piecewise_sequence():
    filt = CausalSignedStickyFilter(num_classes=4)
    labels = torch.tensor([1, 1, 2, 2, 1])
    raw = torch.full((1, len(labels), 4), -20.0)
    raw[0, torch.arange(len(labels)), labels] = 20.0
    event = torch.full((1, len(labels), 3), -20.0)
    event[..., CausalSignedStickyFilter.STAY] = 20.0
    event[0, 2] = torch.tensor([-20.0, -20.0, 20.0])
    event[0, 4] = torch.tensor([20.0, -20.0, -20.0])
    log_q, _ = filt(raw, event)
    assert torch.equal(log_q.argmax(-1)[0], labels)
    assert log_q.dtype == torch.float32
    print("  known signed state sequence    OK")


def test_offline_streaming_equivalence():
    head = build_head()
    x = torch.randn(2, 79, 12)
    with torch.no_grad():
        full = head(x)
    fixed, fixed_cache = chunked(head, x, [16, 16, 16, 16, 15])
    ragged, ragged_cache = chunked(head, x, [1, 7, 3, 22, 5, 41])
    d_fixed = max_output_error(full, fixed)
    d_ragged = max_output_error(full, ragged)
    assert d_fixed < 2e-5, d_fixed
    assert d_ragged < 2e-5, d_ragged
    torch.testing.assert_close(fixed_cache.log_q, full.count_logits[:, -1],
                               atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(ragged_cache.log_q, full.count_logits[:, -1],
                               atol=2e-5, rtol=1e-5)
    assert fixed_cache.log_q.dtype == torch.float32
    print(f"  full == fixed/ragged chunks   OK ({max(d_fixed, d_ragged):.1e})")


def test_no_future_lookahead():
    head = build_head(seed=1)
    x = torch.randn(1, 61, 12)
    changed = x.clone()
    changed[:, 37:] += 100.0 * torch.randn_like(changed[:, 37:])
    with torch.no_grad():
        a, b = head(x), head(changed)
    for xa, xb in zip(a, b):
        torch.testing.assert_close(xa[:, :37], xb[:, :37], atol=2e-5, rtol=1e-5)
    assert max_output_error(tuple(v[:, :37] for v in a),
                            tuple(v[:, :37] for v in b)) < 2e-5
    print("  future perturbation isolation OK")


def test_fp16_inputs_float32_recurrence_and_gradients():
    filt = CausalSignedStickyFilter(num_classes=4)
    raw = torch.randn(2, 257, 4, dtype=torch.float16, requires_grad=True)
    event = torch.randn(2, 257, 3, dtype=torch.float16, requires_grad=True)
    log_q, last = filt(raw, event)
    assert log_q.dtype == torch.float32 and last.dtype == torch.float32
    loss = log_q.square().mean()
    loss.backward()
    assert torch.isfinite(log_q).all() and torch.isfinite(last).all()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert event.grad is not None and torch.isfinite(event.grad).all()
    print("  fp16 input/fp32 recurrence     OK")


def test_cache_validation_and_context_effect():
    head = build_head(seed=2)
    x = torch.randn(1, 25, 12)
    with torch.no_grad():
        _, cache = head.forward_streaming(x[:, :13])
        warm, _ = head.forward_streaming(x[:, 13:], cache)
        cold, _ = head.forward_streaming(x[:, 13:], None)
    assert max_output_error(warm, cold) > 1e-4

    bad = EventStateCache(cache.tcn_caches, cache.log_q.half())
    try:
        head.forward_streaming(x[:, 13:], bad)
    except ValueError:
        pass
    else:
        raise AssertionError("non-float32 q cache must be rejected")
    try:
        head.forward_streaming(x[:, 13:], cache.tcn_caches)
    except ValueError:
        pass
    else:
        raise AssertionError("legacy bare cache list must be rejected")
    print("  cache validation/context       OK")


def test_builder_support():
    config = {
        "model": {
            "encoder": {"type": "mock", "output_dim": 12},
            "head": {
                "type": "tcn_state_ordinal",
                "d_model": 8,
                "dilations": [1, 2],
                "kernel": 3,
                "dropout": 0.0,
                "event_stay_bias": 1.25,
                "state_emission_scale": 0.75,
                "use_state_filter": False,
            },
            "num_count_classes": 4,
        }
    }
    model = build_model(config)
    assert isinstance(model.head, TCNEventStateCountHead)
    assert model.head.state_filter.emission_scale == 0.75
    assert model.head.use_state_filter is False
    print("  build_model config path        OK")


def test_event_aux_public_output_is_raw():
    head = build_head(seed=3)
    head.use_state_filter = False
    x = torch.randn(2, 31, 12)
    with torch.no_grad():
        out = head(x)
    torch.testing.assert_close(out.count_logits, out.raw_count_logits,
                               atol=0, rtol=0)
    print("  event-aux public output raw    OK")


if __name__ == "__main__":
    print("test_event_state_streaming:")
    test_transition_invariants()
    test_known_piecewise_sequence()
    test_offline_streaming_equivalence()
    test_no_future_lookahead()
    test_fp16_inputs_float32_recurrence_and_gradients()
    test_cache_validation_and_context_effect()
    test_builder_support()
    test_event_aux_public_output_is_raw()
    print("ALL PASSED")
