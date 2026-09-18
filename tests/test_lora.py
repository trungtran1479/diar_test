"""Regression tests for src/models/lora.py's LoRAWeightLinear and
WavLMBackbone.apply_lora.

Why a custom implementation instead of `peft`: WavLMAttention.
torch_multi_head_self_attention reads `self.q_proj.weight` directly and
feeds it into `F.multi_head_attention_forward(..., q_proj_weight=...)` --
it never calls `self.q_proj(x)`. A peft-wrapped Linear only overrides
`forward()`, so its LoRA path is never reached there: `.weight` resolves
straight to the frozen base weight, and the whole encoder ends up with NO
gradient path to lora_A/lora_B (reproduced directly: layer-0 output
requires_grad was False despite "successful" peft wrapping).
LoRAWeightLinear fixes this by making `.weight` itself a computed property.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.lora import LoRAWeightLinear  # noqa: E402


def test_weight_property_equals_base_at_init():
    base = nn.Linear(8, 8)
    for p in base.parameters():
        p.requires_grad = False
    lw = LoRAWeightLinear(base, rank=4, alpha=8)
    assert torch.equal(lw.weight, base.weight)


def test_direct_weight_read_path_has_gradient_to_lora_params():
    """Simulates WavLMAttention's call pattern: read `.weight` directly and
    feed it into a functional call, NOT `lw(x)`/`lw.forward(x)`."""
    torch.manual_seed(0)
    base = nn.Linear(8, 8)
    for p in base.parameters():
        p.requires_grad = False
    lw = LoRAWeightLinear(base, rank=4, alpha=8)

    x = torch.randn(3, 8)
    out = torch.nn.functional.linear(x, lw.weight, lw.bias)
    out.sum().backward()

    assert lw.lora_A.grad is not None and torch.isfinite(lw.lora_A.grad).all()
    assert lw.lora_B.grad is not None and torch.isfinite(lw.lora_B.grad).all()
    assert base.weight.grad is None  # frozen


def test_nonzero_lora_b_changes_weight():
    torch.manual_seed(0)
    base = nn.Linear(8, 8)
    for p in base.parameters():
        p.requires_grad = False
    lw = LoRAWeightLinear(base, rank=4, alpha=8)
    with torch.no_grad():
        lw.lora_B.add_(1.0)
    assert not torch.equal(lw.weight, base.weight)


def test_base_layer_params_frozen_on_construction():
    base = nn.Linear(8, 8)
    lw = LoRAWeightLinear(base, rank=4, alpha=8)
    assert not any(p.requires_grad for p in lw.base_layer.parameters())
    assert lw.lora_A.requires_grad and lw.lora_B.requires_grad


def test_lora_params_match_base_dtype():
    """apply_lora() runs after model.to(device)/AMP setup, so a bare
    torch.zeros(...) defaulting to CPU/fp32 would break the very next
    forward with a device or dtype mismatch -- reproduced directly against
    a non-default dtype here without needing a second GPU-backed test."""
    base = nn.Linear(8, 8).to(torch.float64)
    lw = LoRAWeightLinear(base, rank=4, alpha=8)
    assert lw.lora_A.dtype == torch.float64
    assert lw.lora_B.dtype == torch.float64
    assert lw.weight.dtype == torch.float64
