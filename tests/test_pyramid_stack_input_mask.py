"""Parameter-matched fixed stack-input masks for ``GatedPyramidOrdinalHead``.

Mirrors ``test_tcn_stack_input_mask.py``: the mask is an experimental
control, not a smaller architecture.  Every module and parameter must remain
present and identically initialized; only the *projected* feature of a
disabled stack is zeroed before gated fusion (each pyramid stack owns a
biased lateral projection, so masking before it would leak ``silu(bias)``
into the fusion).
"""

import hashlib
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.pyramid_head import GatedPyramidOrdinalHead  # noqa: E402
from src.models import zipcount_v1  # noqa: E402


STACK_DIMS = [12, 16, 20, 24, 20, 16]
D_IN = sum(STACK_DIMS)
PRE012 = [1, 1, 1, 0, 0, 0]


def make(mask=None, seed=1729, gate_mode="global", stage2=False):
    torch.manual_seed(seed)
    return GatedPyramidOrdinalHead(
        d_in=D_IN,
        stack_dims=STACK_DIMS,
        final_dim=None,
        d_model=16,
        dilations=(1, 2, 4, 8, 16, 32),
        kernel=3,
        dropout=0.0,
        gate_mode=gate_mode,
        ordinal_mode="softmax",
        edge_hidden_dim=8,
        use_stage2=stage2,
        stack_input_mask=mask,
    ).eval()


def state_sha(module):
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


def tensor_fields(output):
    return [
        (name, value)
        for name, value in zip(output._fields, output)
        if isinstance(value, torch.Tensor)
    ]


@pytest.mark.parametrize("gate_mode", ["global", "framewise"])
def test_masks_preserve_state_dict_parameters_and_initialization(gate_mode):
    default = make(None, gate_mode=gate_mode)
    all_stacks = make([1] * len(STACK_DIMS), gate_mode=gate_mode)
    pre012 = make(PRE012, gate_mode=gate_mode)

    reference = default.state_dict()
    reference_shapes = {key: tuple(value.shape) for key, value in reference.items()}
    reference_params = sum(parameter.numel() for parameter in default.parameters())
    reference_sha = state_sha(default)

    for head in (all_stacks, pre012):
        assert list(head.state_dict()) == list(reference)
        assert {
            key: tuple(value.shape) for key, value in head.state_dict().items()
        } == reference_shapes
        assert (
            sum(parameter.numel() for parameter in head.parameters())
            == reference_params
        )
        assert state_sha(head) == reference_sha


def test_all_ones_is_exact_default_path():
    default = make(None)
    explicit = make([1] * len(STACK_DIMS))
    x = torch.randn(2, 31, D_IN)
    with torch.no_grad():
        expected = default(x)
        actual = explicit(x)
    for (name, lhs), (_, rhs) in zip(tensor_fields(expected), tensor_fields(actual)):
        assert torch.equal(lhs, rhs), name


@pytest.mark.parametrize("gate_mode", ["global", "framewise"])
def test_masked_perturbations_are_invisible_but_kept_stack_is_not(gate_mode):
    head = make(PRE012, gate_mode=gate_mode)
    x = torch.randn(2, 37, D_IN)
    keep_hi = stack_bounds(2)[1]  # stacks 0-2 kept, 3-5 masked

    masked_change = x.clone()
    masked_change[..., keep_hi:] += torch.randn_like(masked_change[..., keep_hi:])

    kept_change = x.clone()
    kept_change[..., :keep_hi] += torch.randn_like(kept_change[..., :keep_hi])

    with torch.no_grad():
        baseline = head(x)
        masked = head(masked_change)
        kept = head(kept_change)

    for (name, lhs), (_, rhs) in zip(tensor_fields(baseline), tensor_fields(masked)):
        assert torch.equal(lhs, rhs), f"a disabled stack leaked into {name}"
    assert (baseline.count_logits - kept.count_logits).abs().max() > 1.0e-4


def test_masked_norm_and_projection_have_no_data_gradient():
    head = make(PRE012).train()
    x = torch.randn(2, 19, D_IN, requires_grad=True)
    output = head(x)
    sum(value.square().sum() for _, value in tensor_fields(output)).backward()

    keep_hi = stack_bounds(2)[1]
    assert x.grad is not None
    assert torch.count_nonzero(x.grad[..., keep_hi:]) == 0
    assert torch.count_nonzero(x.grad[..., :keep_hi]) > 0

    for index in range(len(STACK_DIMS)):
        norm = head.neck.stack_norms[index]
        projection = head.neck.stack_projections[index]
        if PRE012[index]:
            assert torch.count_nonzero(norm.weight.grad) > 0
            assert torch.count_nonzero(projection.weight.grad) > 0
        else:
            for parameter in (
                norm.weight, norm.bias, projection.weight, projection.bias
            ):
                assert parameter.grad is None \
                    or torch.count_nonzero(parameter.grad) == 0


@pytest.mark.parametrize("gate_mode", ["global", "framewise"])
def test_masked_nonfinite_stack_cannot_poison_output(gate_mode):
    head = make(PRE012, gate_mode=gate_mode)
    x = torch.randn(1, 11, D_IN)
    finite = x.clone()
    keep_hi = stack_bounds(2)[1]
    x[..., keep_hi : keep_hi + STACK_DIMS[3]] = float("nan")
    x[..., keep_hi + STACK_DIMS[3] :] = float("inf")
    with torch.no_grad():
        poisoned = head(x)
        baseline = head(finite)
    for (name, lhs), (_, rhs) in zip(
        tensor_fields(poisoned), tensor_fields(baseline)
    ):
        assert torch.isfinite(lhs).all(), name
        assert torch.equal(lhs, rhs), name


@pytest.mark.parametrize("gate_mode", ["global", "framewise"])
def test_masked_head_streaming_matches_full_forward(gate_mode):
    head = make(PRE012, gate_mode=gate_mode, stage2=True)
    # Give the zero-initialized stage-2 corrections nonzero weights so the
    # equivalence also covers the refinement path.
    with torch.no_grad():
        for correction in (
            head.ordinal_correction,
            head.boundary_correction,
            head.direction_correction,
        ):
            correction.weight.normal_(0.0, 0.05)
            correction.bias.normal_(0.0, 0.05)

    x = torch.randn(1, 73, D_IN)
    with torch.no_grad():
        full = head(x)
        cache = None
        chunks = []
        position = 0
        for size in (16, 7, 1, 32, 17):
            output, cache = head.forward_streaming(
                x[:, position : position + size], cache
            )
            chunks.append(output)
            position += size

    assert position == x.shape[1]
    for index, (name, expected) in enumerate(tensor_fields(full)):
        actual = torch.cat([tensor_fields(c)[index][1] for c in chunks], dim=1)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=2e-5, msg=name)


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


def test_build_model_wires_mask_from_yaml_config(monkeypatch):
    class FakeMultiscaleEncoder(torch.nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()
            self.output_dim = D_IN
            self.stack_dims = list(STACK_DIMS)

        def forward_features(self, features, feature_lens):
            return features, feature_lens

    monkeypatch.setattr(
        zipcount_v1, "StreamingZipformerEncoder", FakeMultiscaleEncoder
    )
    config = {
        "model": {
            "encoder": {"type": "zipformer", "multiscale": True},
            "head": {
                "type": "gated_pyramid_ordinal",
                "d_model": 16,
                "gate_mode": "global",
                "ordinal_mode": "softmax",
                "use_stage2": False,
                "stack_input_mask": PRE012,
            },
            "num_count_classes": 4,
        }
    }
    model = zipcount_v1.build_model(config)
    assert model.head.neck.stack_input_mask == tuple(PRE012)

    config["model"]["head"].pop("stack_input_mask")
    model_default = zipcount_v1.build_model(config)
    assert model_default.head.neck.stack_input_mask == (1,) * len(STACK_DIMS)
