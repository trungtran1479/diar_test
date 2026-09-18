"""Regression tests for WavLMBackbone's partial-unfreeze (unfreeze_last_n).

Before this fix, WavLMBackbone.__init__ accepted no `unfreeze_last_n`
parameter at all, and zipcount_v1.py's wavlm branch of build_model() never
even tried to pass one (unlike the zipformer branch). train.py's
stage2_unfreeze_last_stacks printed "Unfreezing last N stacks" and
proceeded as if it had happened, but `freeze=True` always froze the ENTIRE
WavLM encoder regardless -- a WavLM partial-unfreeze config silently
trained head-only. Uses a tiny randomly-initialized WavLMConfig (not a
real download) so this runs fast and offline.
"""

from __future__ import annotations

import os
import sys

import pytest
import transformers.models.wavlm.modeling_wavlm as wavlm_mod
from transformers import WavLMConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.wavlm_wrapper import WavLMBackbone  # noqa: E402

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


def _trainable_layer_indices(backbone):
    layers = backbone.encoder.encoder.layers
    return [i for i, layer in enumerate(layers) if any(p.requires_grad for p in layer.parameters())]


def test_unfreeze_last_n_zero_is_fully_frozen():
    bb = WavLMBackbone(pretrained_model="fake", freeze=True, collect_layers=False, unfreeze_last_n=0)
    assert not any(p.requires_grad for p in bb.encoder.parameters())


def test_unfreeze_last_2_unfreezes_exactly_last_2_layers():
    bb = WavLMBackbone(pretrained_model="fake", freeze=True, collect_layers=False, unfreeze_last_n=2)
    assert _trainable_layer_indices(bb) == [4, 5]


def test_unfreeze_frontend_stays_frozen():
    """Feature extractor (CNN) and feature projection must stay frozen
    regardless of unfreeze_last_n -- same convention as
    StreamingZipformerEncoder never unfreezing encoder_embed."""
    bb = WavLMBackbone(pretrained_model="fake", freeze=True, collect_layers=False, unfreeze_last_n=6)
    assert not any(p.requires_grad for p in bb.encoder.feature_extractor.parameters())
    assert not any(p.requires_grad for p in bb.encoder.feature_projection.parameters())


def test_unfreeze_last_n_exceeds_layer_count_unfreezes_all_layers():
    bb = WavLMBackbone(pretrained_model="fake", freeze=True, collect_layers=False, unfreeze_last_n=999)
    assert _trainable_layer_indices(bb) == list(range(6))


def test_freeze_false_ignores_unfreeze_last_n_leaves_everything_trainable():
    bb = WavLMBackbone(pretrained_model="fake", freeze=False, collect_layers=False, unfreeze_last_n=0)
    assert all(p.requires_grad for p in bb.encoder.parameters())
