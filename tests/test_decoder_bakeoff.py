"""The decoder bake-off must change only the temporal head and seed."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import runpy
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_SCRIPT = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/run_decoder_bakeoff.py"))
ARM_HEADS = _SCRIPT["ARM_HEADS"]
DEFAULT_BASE = _SCRIPT["DEFAULT_BASE"]
STACK_MASK = _SCRIPT["STACK_MASK"]
assert_matched_budget = _SCRIPT["assert_matched_budget"]
make_config = _SCRIPT["make_config"]
parameter_counts = _SCRIPT["parameter_counts"]


def _base():
    with DEFAULT_BASE.open() as stream:
        return yaml.safe_load(stream)


def test_all_primary_heads_are_parameter_matched():
    base = _base()
    counts = parameter_counts(base, ARM_HEADS)
    assert_matched_budget(counts)
    reference = counts["tcn"]
    assert set(counts) == set(ARM_HEADS)
    assert all(abs(value / reference - 1.0) <= 0.05 for value in counts.values())


def test_arm_generation_preserves_shared_protocol():
    base = _base()
    generated = {
        arm: make_config(base, arm, 1234, DEFAULT_BASE)
        for arm in ARM_HEADS
    }

    reference = copy.deepcopy(generated["tcn"])
    reference.pop("bakeoff")
    reference["model"].pop("head")
    reference["training"].pop("log_dir")

    for arm, config in generated.items():
        shared = copy.deepcopy(config)
        shared.pop("bakeoff")
        shared["model"].pop("head")
        shared["training"].pop("log_dir")
        assert shared == reference, arm
        assert config["model"]["head"]["stack_input_mask"] == STACK_MASK
        assert config["training"]["freeze_backbone_all"] is True
        assert config["training"]["init_backbone_only"] is True
        assert config["training"]["require_backbone_complete"] is True


def test_seed_changes_only_seed_metadata_and_log_directory():
    base = _base()
    first = make_config(base, "ssm", 1234, Path(DEFAULT_BASE))
    second = make_config(base, "ssm", 2345, Path(DEFAULT_BASE))
    for config in (first, second):
        config["training"].pop("seed")
        config["training"].pop("log_dir")
        config["bakeoff"].pop("seed")
    assert first == second
