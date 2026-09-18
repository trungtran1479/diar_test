"""Fast regression tests for the stack-mask graduation runner."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import run_stack_mask_graduation as grad  # noqa: E402


def test_tensor_digest_accepts_scalar_and_empty_tensors():
    tensors = [
        ("scalar", torch.tensor(1.25)),
        ("empty", torch.empty(0, 3)),
    ]
    first = grad.tensor_digest(tensors)
    second = grad.tensor_digest(tensors)
    changed = grad.tensor_digest([
        ("scalar", torch.tensor(1.5)),
        ("empty", torch.empty(0, 3)),
    ])

    assert first == second
    assert first != changed
