"""Parameter-matched fixed stack-input masks for ``TCNCountHead``.

The mask is an experimental control, not a smaller architecture: every
module and parameter must remain present and identically initialized.  It
only zeros selected stack slices after their individual LayerNorms.
"""

import hashlib
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.heads import StackGatedProjection, TCNCountHead  # noqa: E402
from src.models import zipcount_v1  # noqa: E402


STACK_DIMS = [12, 16, 20, 24, 20, 16]
D_IN = sum(STACK_DIMS)
ONLY_STACK_1 = [0, 1, 0, 0, 0, 0]


def make(mask=None, seed=1729):
    torch.manual_seed(seed)
    return TCNCountHead(
        D_IN,
        stack_dims=STACK_DIMS,
        stack_input_mask=mask,
        d_model=32,
        dilations=(1, 2, 4),
        dropout=0.0,
    ).eval()


def state_sha(module):
    """Canonical tensor hash independent of torch.save container metadata."""
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def stack_bounds(index):
    lo = sum(STACK_DIMS[:index])
    return lo, lo + STACK_DIMS[index]


def test_masks_preserve_state_dict_parameters_and_initialization():
    default = make(None)
    all_stacks = make([1] * len(STACK_DIMS))
    only_one = make(ONLY_STACK_1)

    reference = default.state_dict()
    reference_shapes = {key: tuple(value.shape) for key, value in reference.items()}
    reference_params = sum(parameter.numel() for parameter in default.parameters())
    reference_sha = state_sha(default)

    for head in (all_stacks, only_one):
        assert list(head.state_dict()) == list(reference)
        assert {
            key: tuple(value.shape) for key, value in head.state_dict().items()
        } == reference_shapes
        assert sum(parameter.numel() for parameter in head.parameters()) == reference_params
        assert state_sha(head) == reference_sha
        for key, value in head.state_dict().items():
            assert torch.equal(value, reference[key]), key


def test_all_ones_is_exact_default_path():
    default = make(None)
    explicit = make([1] * len(STACK_DIMS))
    x = torch.randn(2, 31, D_IN)
    with torch.no_grad():
        expected = default(x)
        actual = explicit(x)
    for lhs, rhs in zip(expected, actual):
        assert torch.equal(lhs, rhs)


def test_masked_perturbations_are_invisible_but_kept_stack_is_not():
    head = make(ONLY_STACK_1)
    x = torch.randn(2, 37, D_IN)
    keep_lo, keep_hi = stack_bounds(1)

    masked_change = x.clone()
    masked_change[..., :keep_lo] += torch.randn_like(masked_change[..., :keep_lo])
    masked_change[..., keep_hi:] += torch.randn_like(masked_change[..., keep_hi:])

    kept_change = x.clone()
    kept_change[..., keep_lo:keep_hi] += torch.randn_like(
        kept_change[..., keep_lo:keep_hi])

    with torch.no_grad():
        baseline = head(x)
        masked = head(masked_change)
        kept = head(kept_change)

    for lhs, rhs in zip(baseline, masked):
        assert torch.equal(lhs, rhs), "a disabled stack leaked into the output"
    assert (baseline[0] - kept[0]).abs().max() > 1.0e-4


def test_masked_layernorm_and_projection_columns_have_zero_data_gradient():
    head = make(ONLY_STACK_1)
    x = torch.randn(2, 19, D_IN, requires_grad=True)
    sum(tensor.square().sum() for tensor in head(x)).backward()

    keep_lo, keep_hi = stack_bounds(1)
    assert x.grad is not None
    assert torch.count_nonzero(x.grad[..., :keep_lo]) == 0
    assert torch.count_nonzero(x.grad[..., keep_hi:]) == 0
    assert torch.count_nonzero(x.grad[..., keep_lo:keep_hi]) > 0

    for index, norm in enumerate(head.fusion.norms):
        if index == 1:
            assert torch.count_nonzero(norm.weight.grad) > 0
            assert torch.count_nonzero(norm.bias.grad) > 0
        else:
            assert norm.weight.grad is None \
                or torch.count_nonzero(norm.weight.grad) == 0
            assert norm.bias.grad is None \
                or torch.count_nonzero(norm.bias.grad) == 0

    projection_grad = head.fusion.proj.weight.grad
    assert projection_grad is not None
    assert torch.count_nonzero(projection_grad[:, :keep_lo]) == 0
    assert torch.count_nonzero(projection_grad[:, keep_hi:]) == 0
    assert torch.count_nonzero(projection_grad[:, keep_lo:keep_hi]) > 0


def test_masked_nonfinite_stack_cannot_poison_output():
    head = make(ONLY_STACK_1)
    x = torch.randn(1, 11, D_IN)
    keep_lo, keep_hi = stack_bounds(1)
    x[..., :keep_lo] = float("nan")
    x[..., keep_hi:] = float("inf")
    with torch.no_grad():
        output = head(x)
    assert all(torch.isfinite(tensor).all() for tensor in output)


def test_masked_head_streaming_matches_full_forward():
    head = make([1, 1, 1, 0, 0, 0])
    x = torch.randn(1, 73, D_IN)
    with torch.no_grad():
        full = head(x)
        caches = None
        chunks = []
        position = 0
        for size in (16, 7, 1, 32, 17):
            output, caches = head.forward_streaming(
                x[:, position:position + size], caches)
            chunks.append(output)
            position += size

    assert position == x.shape[1]
    for index, expected in enumerate(full):
        actual = torch.cat([chunk[index] for chunk in chunks], dim=1)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=2.0e-6)


@pytest.mark.parametrize(
    "mask",
    (
        [],
        [1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 2, 0, 0, 0],
        [1, 0, 0.5, 0, 0, 0],
        [1, 0, "1", 0, 0, 0],
    ),
)
def test_mask_validation(mask):
    with pytest.raises(ValueError):
        make(mask)


def test_mask_requires_stack_dims_and_is_mutually_exclusive_with_indices():
    with pytest.raises(ValueError, match="requires stack_dims"):
        TCNCountHead(D_IN, stack_input_mask=ONLY_STACK_1)
    with pytest.raises(ValueError, match="cannot be combined"):
        TCNCountHead(
            D_IN,
            stack_dims=STACK_DIMS,
            stack_indices=[1],
            stack_input_mask=ONLY_STACK_1,
        )


def test_projection_validates_direct_use_too():
    with pytest.raises(ValueError):
        StackGatedProjection(STACK_DIMS, stack_input_mask=[0] * len(STACK_DIMS))


def test_build_model_wires_mask_from_yaml_config(monkeypatch):
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
            "encoder": {
                "type": "zipformer",
                "multiscale": True,
            },
            "head": {
                "type": "tcn_ordinal",
                "d_model": 32,
                "dilations": [1, 2],
                "dropout": 0.0,
                "stack_input_mask": ONLY_STACK_1,
            },
        },
    }
    model = zipcount_v1.build_model(config)
    assert isinstance(model.head, TCNCountHead)
    assert model.head.fusion.stack_input_mask == tuple(ONLY_STACK_1)
