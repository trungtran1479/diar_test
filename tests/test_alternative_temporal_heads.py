"""Controls for temporal decoders used in the decoder bake-off.

The most important contract is chunk invariance: in evaluation mode a causal
head must return the same frame logits whether a recording is passed at once
or split into arbitrary streaming chunks with its cache carried forward.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import zipcount_v1  # noqa: E402
from src.models.heads import (  # noqa: E402
    CausalAttentionCache,
    CausalAttentionCountHead,
    CausalGRUCountHead,
    CausalSSMCountHead,
)


STACK_DIMS = [5, 7, 4]
D_IN = sum(STACK_DIMS)


def _make(kind: str):
    common = dict(
        d_in=D_IN,
        num_classes=4,
        d_model=12,
        dropout=0.0,
        stack_dims=STACK_DIMS,
        stack_input_mask=[1, 1, 1],
    )
    torch.manual_seed(17)
    if kind == "gru":
        return CausalGRUCountHead(**common, num_layers=2).eval()
    if kind == "ssm":
        return CausalSSMCountHead(**common, state_dim=10).eval()
    if kind == "attention":
        return CausalAttentionCountHead(
            **common, num_heads=3, max_context=9).eval()
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["gru", "ssm", "attention"])
def test_full_equals_arbitrary_streaming_chunks(kind: str):
    head = _make(kind)
    x = torch.randn(2, 37, D_IN)
    sizes = [1, 8, 3, 11, 2, 12]

    with torch.no_grad():
        expected = head(x)
        cache = None
        parts = [[], [], []]
        position = 0
        for size in sizes:
            output, cache = head.forward_streaming(
                x[:, position:position + size], cache)
            for index, value in enumerate(output):
                parts[index].append(value)
            position += size
        actual = tuple(torch.cat(values, dim=1) for values in parts)

    assert position == x.shape[1]
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("kind", ["gru", "ssm", "attention"])
def test_future_frames_cannot_change_past(kind: str):
    head = _make(kind)
    x = torch.randn(1, 31, D_IN)
    changed = x.clone()
    changed[:, 19:] += 20.0 * torch.randn_like(changed[:, 19:])

    with torch.no_grad():
        before = head(x)[0]
        after = head(changed)[0]
    torch.testing.assert_close(
        before[:, :19], after[:, :19], atol=2e-6, rtol=2e-6)


def test_attention_cache_is_bounded():
    head = _make("attention")
    x = torch.randn(2, 41, D_IN)
    with torch.no_grad():
        _, cache = head.forward_streaming(x[:, :17])
        _, cache = head.forward_streaming(x[:, 17:])
    assert isinstance(cache, CausalAttentionCache)
    assert len(cache.layers) == head.num_layers
    for layer in cache.layers:
        assert layer.key.shape == (2, 3, head.max_context - 1, 4)
        assert layer.value.shape == layer.key.shape


@pytest.mark.parametrize("kind", ["gru", "ssm", "attention"])
def test_streaming_cache_changes_later_predictions(kind: str):
    head = _make(kind)
    x = torch.randn(1, 25, D_IN)
    with torch.no_grad():
        _, cache = head.forward_streaming(x[:, :13])
        warm, _ = head.forward_streaming(x[:, 13:], cache)
        cold, _ = head.forward_streaming(x[:, 13:], None)
    assert (warm[0] - cold[0]).abs().max() > 1e-5


@pytest.mark.parametrize("kind", ["gru", "ssm", "attention"])
def test_all_trainable_parameters_receive_finite_gradients(kind: str):
    head = _make(kind).train()
    output = head(torch.randn(2, 13, D_IN))
    loss = sum(value.square().mean() for value in output)
    loss.backward()
    missing = [
        name for name, parameter in head.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    nonfinite = [
        name for name, parameter in head.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    assert not missing
    assert not nonfinite


@pytest.mark.parametrize(
    ("head_type", "expected"),
    [
        ("gru_ordinal", CausalGRUCountHead),
        ("ssm_ordinal", CausalSSMCountHead),
        ("attention_ordinal", CausalAttentionCountHead),
    ],
)
def test_model_factory_wires_alternative_heads(monkeypatch, head_type, expected):
    class FakeMultiscaleEncoder(torch.nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()
            self.output_dim = D_IN
            self.stack_dims = list(STACK_DIMS)

        def forward_features(self, features, feature_lens):
            return features, feature_lens

    monkeypatch.setattr(
        zipcount_v1, "StreamingZipformerEncoder", FakeMultiscaleEncoder)
    config = {
        "model": {
            "encoder": {"type": "zipformer", "multiscale": True},
            "head": {
                "type": head_type,
                "d_model": 12,
                "num_layers": 1,
                "state_dim": 10,
                "num_heads": 3,
                "max_context": 9,
                "dropout": 0.0,
                "stack_input_mask": [1, 1, 0],
            },
        },
        "training": {"seed": 17},
    }

    model = zipcount_v1.build_model(config)
    assert isinstance(model.head, expected)
    assert model.head.fusion.stack_input_mask == (1, 1, 0)


def test_invalid_attention_configuration_and_cache_fail_closed():
    with pytest.raises(ValueError, match="divisible"):
        CausalAttentionCountHead(D_IN, d_model=10, num_heads=3)
    with pytest.raises(ValueError, match="max_context"):
        CausalAttentionCountHead(D_IN, d_model=12, num_heads=3, max_context=0)

    head = _make("attention")
    with pytest.raises(ValueError, match="attention cache"):
        head.forward_streaming(torch.randn(1, 3, D_IN), [])
