"""Regression test: a partially-frozen backbone must keep frozen layers in
eval() (dropout off) while unfrozen layers are in train() (dropout on).

Before this fix, train.py's set_train_mode() special-cased only the FULLY
frozen backbone (`model.backbone.eval()` iff no param anywhere requires
grad); a partial unfreeze (unfreeze_last_n>0) left `backbone_frozen=False`,
so plain `model.train()` put the WHOLE backbone -- including its frozen
early layers -- into train mode. A head_only vs unfreeze_last_n comparison
would then differ in both adaptation AND stochastic-backbone noise in the
frozen layers, confounding the ablation.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.train_mode import set_backbone_partial_train_mode  # noqa: E402


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return self.dropout(self.linear(x))


class _TinyBackbone(nn.Module):
    def __init__(self, n_layers=4):
        super().__init__()
        self.frontend = nn.Linear(4, 4)  # analogous to CNN feature extractor
        self.layers = nn.ModuleList(_Layer() for _ in range(n_layers))

    def freeze_all_but_last(self, n):
        for p in self.parameters():
            p.requires_grad = False
        for layer in self.layers[len(self.layers) - n:]:
            for p in layer.parameters():
                p.requires_grad = True


def test_fully_frozen_backbone_all_eval():
    bb = _TinyBackbone()
    bb.freeze_all_but_last(0)
    bb.train()  # simulate the naive model.train() that would otherwise run first
    set_backbone_partial_train_mode(bb)
    assert bb.frontend.training is False
    assert all(not layer.training for layer in bb.layers)


def test_partial_unfreeze_only_unfrozen_layers_train():
    bb = _TinyBackbone(n_layers=4)
    bb.freeze_all_but_last(2)
    bb.train()
    set_backbone_partial_train_mode(bb)
    assert bb.frontend.training is False
    assert [layer.training for layer in bb.layers] == [False, False, True, True]
    # dropout submodules of unfrozen layers must also be in train mode
    assert bb.layers[2].dropout.training is True
    assert bb.layers[3].dropout.training is True
    # dropout submodules of frozen layers must stay in eval mode
    assert bb.layers[0].dropout.training is False
    assert bb.layers[1].dropout.training is False


def test_fully_unfrozen_backbone_all_train():
    bb = _TinyBackbone(n_layers=3)
    for p in bb.parameters():
        p.requires_grad = True
    bb.eval()  # start from eval to prove the function turns things back on
    set_backbone_partial_train_mode(bb)
    assert bb.frontend.training is True
    assert all(layer.training for layer in bb.layers)


def test_dropout_actually_differs_between_frozen_and_unfrozen_layers():
    """End-to-end check: with a fixed seed, a frozen layer's output must be
    deterministic across repeated forward passes (eval-mode dropout is a
    no-op), while an unfrozen layer's output must vary (train-mode dropout
    is stochastic)."""
    torch.manual_seed(0)
    bb = _TinyBackbone(n_layers=2)
    bb.freeze_all_but_last(1)
    bb.train()
    set_backbone_partial_train_mode(bb)

    x = torch.ones(1, 4)
    torch.manual_seed(1)
    frozen_out_a = bb.layers[0](x).clone()
    torch.manual_seed(2)
    frozen_out_b = bb.layers[0](x).clone()
    assert torch.allclose(frozen_out_a, frozen_out_b)

    torch.manual_seed(1)
    unfrozen_out_a = bb.layers[1](x).clone()
    torch.manual_seed(2)
    unfrozen_out_b = bb.layers[1](x).clone()
    assert not torch.allclose(unfrozen_out_a, unfrozen_out_b)
