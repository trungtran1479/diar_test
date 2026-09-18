"""Tests for legacy and encoder-learned Zipformer stack alignment."""

from __future__ import annotations

import inspect
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.zipformer_wrapper import StreamingZipformerEncoder  # noqa: E402


class _LearnedPairDownsample(nn.Module):
    """Channel-agnostic stand-in for Zipformer's SimpleDownsample."""

    def __init__(self, weights=(0.1, 0.9)):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(weights).log())
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        length, batch, channels = x.shape
        out_length = (length + 1) // 2
        pad = 2 * out_length - length
        if pad:
            x = torch.cat([x, x[-1:].expand(pad, batch, channels)], dim=0)
        weights = self.bias.softmax(0).view(1, 2, 1, 1)
        return (x.reshape(out_length, 2, batch, channels) * weights).sum(1)


class _FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample_output = _LearnedPairDownsample()


def _bare_wrapper(
    stack_dims,
    *,
    alignment: str,
    include_final: bool = False,
) -> StreamingZipformerEncoder:
    # Avoid constructing the 280 MB recipe model: _hypercolumn only needs
    # these explicitly listed fields.
    wrapper = StreamingZipformerEncoder.__new__(StreamingZipformerEncoder)
    nn.Module.__init__(wrapper)
    wrapper.encoder = _FakeEncoder()
    wrapper.stack_dims = list(stack_dims)
    wrapper.multiscale_alignment = alignment
    wrapper.multiscale_include_final = include_final
    wrapper._stack_outputs = []
    return wrapper


def _stack_inputs(stack_dims, length=7, batch=2):
    pieces = []
    for stack, dim in enumerate(stack_dims):
        base = torch.arange(length, dtype=torch.float32).view(length, 1, 1)
        channel = torch.arange(dim, dtype=torch.float32).view(1, 1, dim)
        pieces.append((100 * stack + 10 * base + channel).expand(-1, batch, -1))
    return pieces


def test_constructor_defaults_preserve_legacy_hypercolumn():
    signature = inspect.signature(StreamingZipformerEncoder.__init__)
    assert signature.parameters["multiscale_alignment"].default == "average"
    assert signature.parameters["multiscale_include_final"].default is False


def test_average_alignment_is_bitwise_the_legacy_operation():
    dims = (2, 3, 1, 4, 2, 3)
    wrapper = _bare_wrapper(dims, alignment="average")
    wrapper._stack_outputs = _stack_inputs(dims)

    actual = wrapper._hypercolumn(t_enc=4)
    expected = torch.cat(
        [
            F.avg_pool1d(
                stack.permute(1, 2, 0), 2, stride=2, ceil_mode=True
            ).transpose(1, 2)
            for stack in wrapper._stack_outputs
        ],
        dim=-1,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert wrapper.encoder.downsample_output.calls == 0


def test_encoder_learned_alignment_uses_exact_shared_operator_per_stack():
    dims = (2, 3, 1, 4, 2, 3)
    wrapper = _bare_wrapper(dims, alignment="encoder_learned")
    wrapper._stack_outputs = _stack_inputs(dims)
    original = [stack.clone() for stack in wrapper._stack_outputs]

    actual = wrapper._hypercolumn(t_enc=4)
    expected = torch.cat(
        [
            wrapper.encoder.downsample_output(stack).transpose(0, 1)
            for stack in original
        ],
        dim=-1,
    )
    # Reset the direct expected-computation calls from the semantic count.
    assert wrapper.encoder.downsample_output.calls == 12
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    legacy = torch.cat(
        [
            F.avg_pool1d(
                stack.permute(1, 2, 0), 2, stride=2, ceil_mode=True
            ).transpose(1, 2)
            for stack in original
        ],
        dim=-1,
    )
    assert (actual - legacy).abs().max() > 1.0

    # The encoder's learned temporal weights must remain on the autograd path.
    actual.square().mean().backward()
    grad = wrapper.encoder.downsample_output.bias.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_actual_final_encoder_output_is_appended_unchanged():
    dims = (1, 1, 1, 1, 1, 1)
    wrapper = _bare_wrapper(
        dims, alignment="encoder_learned", include_final=True
    )
    wrapper._stack_outputs = _stack_inputs(dims, length=8, batch=1)
    final = torch.randn(1, 4, 5)
    actual = wrapper._hypercolumn(t_enc=4, final_output=final)
    assert actual.shape == (1, 4, sum(dims) + 5)
    torch.testing.assert_close(actual[..., -5:], final, atol=0, rtol=0)

