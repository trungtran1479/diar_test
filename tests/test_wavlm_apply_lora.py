"""Regression tests for WavLMBackbone.apply_lora against a real (tiny)
WavLM model -- complements tests/test_lora.py's isolated LoRAWeightLinear
tests with the full wiring (layer selection, invariant assertions, gradient
flow end-to-end through forward_features).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import transformers.models.wavlm.modeling_wavlm as wavlm_mod
from transformers import WavLMConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.wavlm_wrapper import WavLMBackbone  # noqa: E402
from src.utils.train_mode import set_backbone_partial_train_mode  # noqa: E402

_TINY_CFG = WavLMConfig(
    hidden_size=32,
    num_hidden_layers=6,
    num_attention_heads=4,
    intermediate_size=64,
    conv_dim=(32,) * 7,
    conv_stride=(5, 2, 2, 2, 2, 2, 2),
    conv_kernel=(10, 3, 3, 3, 3, 2, 2),
)


@pytest.fixture(autouse=True)
def _mock_from_pretrained(monkeypatch):
    monkeypatch.setattr(
        wavlm_mod.WavLMModel,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: wavlm_mod.WavLMModel(_TINY_CFG)),
    )


def _build():
    return WavLMBackbone(pretrained_model="fake", freeze=True, collect_layers=False, unfreeze_last_n=0)


def test_apply_lora_wraps_exact_module_count():
    bb = _build()
    bb.apply_lora(target_modules=["q_proj", "v_proj"], layer_indices=list(range(6)), rank=4, alpha=8)
    trainable = [n for n, p in bb.encoder.named_parameters() if p.requires_grad]
    assert len(trainable) == 6 * 2 * 2  # 6 layers x 2 target_modules x (lora_A + lora_B)


def test_apply_lora_partial_layers():
    bb = _build()
    bb.apply_lora(target_modules=["q_proj"], layer_indices=[2, 4], rank=4, alpha=8)
    trainable = [n for n, p in bb.encoder.named_parameters() if p.requires_grad]
    assert len(trainable) == 2 * 1 * 2
    for i in (0, 1, 3, 5):
        attn = bb.encoder.encoder.layers[i].attention
        assert not hasattr(attn.q_proj, "lora_A")


def test_apply_lora_rejects_already_trainable_backbone():
    bb = WavLMBackbone(pretrained_model="fake", freeze=True, collect_layers=False, unfreeze_last_n=2)
    with pytest.raises(ValueError, match="fully frozen"):
        bb.apply_lora(target_modules=["q_proj"], layer_indices=[0], rank=4, alpha=8)


def test_apply_lora_rejects_out_of_range_layer_indices():
    bb = _build()
    with pytest.raises(ValueError, match="out of range"):
        bb.apply_lora(target_modules=["q_proj"], layer_indices=[99], rank=4, alpha=8)


def test_gradient_flows_end_to_end_through_forward_features():
    torch.manual_seed(0)
    bb = _build()
    bb.apply_lora(target_modules=["q_proj", "v_proj"], layer_indices=list(range(6)), rank=4, alpha=8)
    bb.train()
    set_backbone_partial_train_mode(bb)

    feats = torch.randn(2, 16000)
    flens = torch.tensor([16000, 16000])
    h, _ = bb.forward_features(feats, flens)
    assert h.requires_grad
    h.sum().backward()

    trainable = [p for n, p in bb.encoder.named_parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
    frozen = [p for n, p in bb.encoder.named_parameters() if not p.requires_grad]
    assert all(p.grad is None for p in frozen)


def test_b_zero_is_noop_and_wrapped_linears_are_reachable():
    """Two-directional check mirroring train.py's own invariant: B=0 must
    be an exact no-op through the REAL forward path, AND perturbing B must
    change the output -- catching the exact failure class peft had
    (wrapped modules present but never reached by the real forward call)."""
    torch.manual_seed(0)
    bb = _build()
    bb.apply_lora(target_modules=["q_proj", "v_proj"], layer_indices=list(range(6)), rank=4, alpha=8)
    bb.eval()

    feats = torch.randn(1, 16000)
    flens = torch.tensor([16000])
    wrapped = [
        getattr(bb.encoder.encoder.layers[i].attention, name)
        for i in range(6) for name in ("q_proj", "v_proj")
    ]
    with torch.no_grad():
        h0, _ = bb.forward_features(feats, flens)
        for w in wrapped:
            w.lora_B.add_(1.0)
        h1, _ = bb.forward_features(feats, flens)
        for w in wrapped:
            w.lora_B.zero_()
        h2, _ = bb.forward_features(feats, flens)

    assert torch.equal(h0, h2)  # exact no-op once reset
    assert (h0 - h1).abs().max().item() > 1e-3  # perturbation actually reaches the output
