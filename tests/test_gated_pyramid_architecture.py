"""Regression tests for the gated temporal-pyramid ordinal decoder.

These tests intentionally exercise trained-state conditions (non-zero
LayerNorm biases and non-zero stage-2 corrections).  Freshly initialized
models can otherwise make a broken cache implementation look correct because
stage 2 starts as an exact no-op.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, Sequence

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.pyramid_head import (  # noqa: E402
    GatedPyramidCache,
    GatedPyramidOrdinalHead,
)


STACK_DIMS = (2, 3, 4, 5, 4, 3)


def _build_head(
    *,
    seed: int = 0,
    mode: str = "corn",
    stage2: bool = True,
    final_dim: int | None = 5,
    framewise_gate: bool = True,
) -> GatedPyramidOrdinalHead:
    torch.manual_seed(seed)
    d_in = sum(STACK_DIMS) + (final_dim or 0)
    head = GatedPyramidOrdinalHead(
        d_in=d_in,
        stack_dims=STACK_DIMS,
        final_dim=final_dim,
        d_model=8,
        dilations=(1, 2, 4, 8, 16, 32),
        kernel=3,
        dropout=0.0,
        use_framewise_gate=framewise_gate,
        ordinal_mode=mode,
        edge_hidden_dim=6,
        use_stage2=stage2,
    ).eval()
    return head


def _make_stage2_observable(head: GatedPyramidOrdinalHead) -> None:
    """Move the zero-init refinement branch to a trained-like state."""

    if not head.use_stage2:
        return
    with torch.no_grad():
        for block in list(head.stage1_blocks) + list(head.stage2_blocks):
            torch.nn.init.normal_(block.norm.bias, std=0.1)
        for layer in (
            head.ordinal_correction,
            head.boundary_correction,
            head.direction_correction,
        ):
            assert layer is not None
            torch.nn.init.normal_(layer.weight, std=0.05)
            torch.nn.init.normal_(layer.bias, std=0.02)


def _tensor_fields(output) -> Iterable[tuple[str, torch.Tensor]]:
    for name, value in zip(output._fields, output):
        if torch.is_tensor(value):
            yield name, value


@pytest.mark.parametrize("mode", ["corn", "softmax"])
@pytest.mark.parametrize("stage2", [False, True])
def test_output_contract_shapes_and_probabilities(mode: str, stage2: bool):
    head = _build_head(mode=mode, stage2=stage2)
    x = torch.randn(2, 19, head.d_in)
    output = head(x)

    assert output.count_logits.shape == (2, 19, 4)
    assert output.vad_logits.shape == (2, 19, 1)
    assert output.overlap_logits.shape == (2, 19, 1)
    assert output.stage1_count_logits.shape == (2, 19, 4)
    assert output.count_log_probs.shape == (2, 19, 4)
    assert output.stage1_count_log_probs.shape == (2, 19, 4)
    ordinal_width = 3 if mode == "corn" else 4
    assert output.ordinal_logits.shape == (2, 19, ordinal_width)
    assert output.stage1_ordinal_logits.shape == (2, 19, ordinal_width)
    assert output.boundary_logits.shape == (2, 19, 1)
    assert output.direction_logits.shape == (2, 19, 1)
    assert output.stage1_boundary_logits.shape == (2, 19, 1)
    assert output.stage1_direction_logits.shape == (2, 19, 1)
    assert output.fusion_gates.shape == (2, 19, 6)

    probs = output.count_log_probs.exp()
    assert torch.isfinite(probs).all()
    assert (probs >= 0).all()
    torch.testing.assert_close(
        probs.sum(-1), torch.ones(2, 19), atol=2e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        output.fusion_gates.sum(-1),
        torch.ones(2, 19),
        atol=2e-6,
        rtol=1e-6,
    )
    assert (output.fusion_gates >= 0).all()

    # Cumulative risks derived from any valid p(0..3) must be monotone.
    q1 = probs[..., 1:].sum(-1)
    q2 = probs[..., 2:].sum(-1)
    q3 = probs[..., 3]
    assert torch.all(q1 + 1e-7 >= q2)
    assert torch.all(q2 + 1e-7 >= q3)

    vad = torch.sigmoid(output.vad_logits.squeeze(-1))
    overlap = torch.sigmoid(output.overlap_logits.squeeze(-1))
    torch.testing.assert_close(vad, q1, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(overlap, q2, atol=2e-6, rtol=1e-5)


def test_corn_conditional_logits_have_exact_cumulative_semantics():
    head = _build_head(mode="corn", stage2=False)
    conditional = torch.tensor(
        [[[2.0, -0.5, 1.25], [-3.0, 4.0, -2.0]]], dtype=torch.float32
    )
    _, logp = head.stage1_ordinal.decode(conditional)
    p = logp.exp()
    cond = conditional.sigmoid()
    expected_q = torch.stack(
        [cond[..., 0], cond[..., 0] * cond[..., 1], cond.prod(-1)], dim=-1
    )
    actual_q = torch.stack(
        [p[..., 1:].sum(-1), p[..., 2:].sum(-1), p[..., 3]], dim=-1
    )
    torch.testing.assert_close(actual_q, expected_q, atol=2e-6, rtol=1e-6)


def test_stage2_is_an_exact_noop_at_initialization():
    head = _build_head(mode="corn", stage2=True)
    output = head(torch.randn(2, 23, head.d_in))
    torch.testing.assert_close(
        output.ordinal_logits, output.stage1_ordinal_logits, atol=0, rtol=0
    )
    torch.testing.assert_close(
        output.count_log_probs, output.stage1_count_log_probs, atol=0, rtol=0
    )
    torch.testing.assert_close(
        output.boundary_logits, output.stage1_boundary_logits, atol=0, rtol=0
    )
    torch.testing.assert_close(
        output.direction_logits, output.stage1_direction_logits, atol=0, rtol=0
    )


@pytest.mark.parametrize("stage2", [False, True])
def test_future_perturbation_cannot_change_past(stage2: bool):
    head = _build_head(seed=3, stage2=stage2)
    _make_stage2_observable(head)
    x = torch.randn(1, 67, head.d_in)
    changed = x.clone()
    changed[:, 41:] += 30.0 * torch.randn_like(changed[:, 41:])
    with torch.no_grad():
        a = head(x)
        b = head(changed)
    for name, xa in _tensor_fields(a):
        xb = getattr(b, name)
        torch.testing.assert_close(
            xa[:, :41], xb[:, :41], atol=3e-5, rtol=2e-5, msg=name
        )


def _chunked(
    head: GatedPyramidOrdinalHead, x: torch.Tensor, sizes: Sequence[int]
):
    cache = None
    pieces = {name: [] for name in head(x[:, :1])._fields}
    pos = 0
    with torch.no_grad():
        for size in sizes:
            output, cache = head.forward_streaming(x[:, pos : pos + size], cache)
            for name, value in _tensor_fields(output):
                pieces[name].append(value)
            pos += size
    assert pos == x.shape[1]
    assert cache is not None
    return {
        name: torch.cat(values, dim=1)
        for name, values in pieces.items()
        if values
    }, cache


@pytest.mark.parametrize(
    "sizes",
    [
        (16, 16, 16, 16, 15),
        (1, 7, 3, 22, 5, 41),
    ],
)
def test_full_equals_chunked_after_stage2_has_trained(sizes: Sequence[int]):
    head = _build_head(seed=5, stage2=True)
    _make_stage2_observable(head)
    x = torch.randn(2, 79, head.d_in)
    with torch.no_grad():
        full = head(x)
    chunked, cache = _chunked(head, x, sizes)

    for name, expected in _tensor_fields(full):
        torch.testing.assert_close(
            chunked[name], expected, atol=3e-5, rtol=2e-5, msg=name
        )
    assert len(cache.stage1_caches) == 6
    assert len(cache.stage2_caches) == 6
    assert cache.prev_edge_hidden.shape == (2, 1, head.d_model)


def test_streaming_cache_is_nontrivial_and_validated():
    head = _build_head(seed=7, stage2=True)
    _make_stage2_observable(head)
    x = torch.randn(1, 73, head.d_in)
    with torch.no_grad():
        _, warm_cache = head.forward_streaming(x[:, :57], None)
        warm, _ = head.forward_streaming(x[:, 57:], warm_cache)
        cold, _ = head.forward_streaming(x[:, 57:], None)
    differences = [
        (value - getattr(cold, name)).abs().max().item()
        for name, value in _tensor_fields(warm)
    ]
    assert max(differences) > 1e-4, "cache has no observable context effect"

    bad = GatedPyramidCache(
        stage1_caches=warm_cache.stage1_caches[:-1],
        stage2_caches=warm_cache.stage2_caches,
        prev_edge_hidden=warm_cache.prev_edge_hidden,
    )
    with pytest.raises(ValueError):
        head.forward_streaming(x[:, 57:], bad)

    bad = GatedPyramidCache(
        stage1_caches=warm_cache.stage1_caches,
        stage2_caches=warm_cache.stage2_caches,
        prev_edge_hidden=warm_cache.prev_edge_hidden[:, :, :-1],
    )
    with pytest.raises(ValueError):
        head.forward_streaming(x[:, 57:], bad)


def test_framewise_gate_reacts_to_frame_content():
    head = _build_head(seed=11, stage2=False, final_dim=None)
    x = torch.randn(1, 13, head.d_in)
    with torch.no_grad():
        gates = head(x).fusion_gates
    assert (gates[:, 1:] - gates[:, :-1]).abs().max() > 1e-5


def test_nonframewise_fusion_has_no_dead_trainable_gate_parameters():
    """The ablation path must not silently add optimizer-only parameters."""

    head = _build_head(
        seed=12, stage2=False, final_dim=None, framewise_gate=False
    )
    dead = [
        name
        for name, parameter in head.named_parameters()
        if "gate_projections" in name and parameter.requires_grad
    ]
    assert not dead, f"unused gate parameters remain trainable: {dead}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fp16 unavailable")
def test_fp16_forward_and_gradient_are_finite():
    head = _build_head(seed=13, stage2=True).cuda().half().train()
    _make_stage2_observable(head)
    x = torch.randn(2, 31, head.d_in, device="cuda", dtype=torch.float16)
    output = head(x)
    # Main and auxiliary branches must all receive a finite gradient signal.
    loss = (
        output.count_log_probs.float().square().mean()
        + output.stage1_count_log_probs.float().square().mean()
        + output.boundary_logits.float().square().mean()
        + output.direction_logits.float().square().mean()
    )
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
