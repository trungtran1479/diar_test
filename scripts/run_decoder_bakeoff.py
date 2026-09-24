#!/usr/bin/env python3
"""Prepare or execute a parameter-matched temporal-decoder bake-off.

All arms keep the same frozen Zipformer backbone, pre012 hypercolumn mask,
192-dimensional fusion, output head, loss, data and optimizer schedule.  Only
the temporal decoder changes.  Running without ``--execute`` writes the exact
YAML files and prints commands; it does not start a long training job.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, Iterable, List

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_BASE = REPO_ROOT / "artifacts/stack_mask_graduation/stackgrad_pre012_s1234.yaml"
# The original r8 checkpoint is no longer in this snapshot.  This retained
# paper checkpoint contains the same r8 backbone (it was loaded then frozen),
# and train.py's init_backbone_only path deliberately ignores its TCN head.
DEFAULT_INIT = REPO_ROOT / "logs/stackgrad_pre012_s3456/step3000.pt"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/decoder_bakeoff"
SEEDS = (1234, 2345, 3456)
STACK_MASK = [1, 1, 1, 0, 0, 0]


# Width/depth choices are parameter matched around the TCN-192 control.  The
# audit below fails if a future edit moves an arm outside a ±5% budget.
ARM_HEADS: Dict[str, dict] = {
    "tcn": {
        "type": "tcn_ordinal",
        "d_model": 192,
        "dilations": [1, 2, 4, 8, 16, 32],
        "kernel": 3,
        "dropout": 0.1,
    },
    "deformable": {
        "type": "deformable_ordinal",
        "d_model": 192,
        "num_layers": 4,
        "num_points": 4,
        "num_groups": 4,
        "max_offset": 16,
        "dropout": 0.1,
    },
    "gru": {
        "type": "gru_ordinal",
        "d_model": 192,
        "num_layers": 4,
        "dropout": 0.1,
    },
    "ssm": {
        "type": "ssm_ordinal",
        "d_model": 192,
        "state_dim": 384,
        "num_layers": 4,
        "dropout": 0.1,
    },
    "attention": {
        "type": "attention_ordinal",
        "d_model": 192,
        "num_heads": 4,
        "num_layers": 3,
        "max_context": 128,
        "ffn_multiplier": 2,
        "dropout": 0.1,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_int_list(value) -> List[int]:
    if isinstance(value, str):
        return [int(item) for item in value.split(",")]
    return [int(item) for item in value]


def make_config(base: dict, arm: str, seed: int, base_path: Path) -> dict:
    if arm not in ARM_HEADS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARM_HEADS)}")
    config = copy.deepcopy(base)
    encoder = config["model"]["encoder"]
    if not bool(encoder.get("multiscale", False)):
        raise ValueError("decoder bake-off requires encoder.multiscale=true")

    head = copy.deepcopy(ARM_HEADS[arm])
    head["stack_input_mask"] = list(STACK_MASK)
    head["renormalize_active_gate"] = False
    config["model"]["head"] = head
    config["training"].update({
        "seed": int(seed),
        "stage": "stage1_freeze_backbone",
        "freeze_backbone_all": True,
        "init_backbone_only": True,
        "require_backbone_complete": True,
        "reset_rng_after_init": True,
        "log_dir": f"logs/decoder_bakeoff/{arm}_s{seed}",
    })
    config["bakeoff"] = {
        "arm": arm,
        "seed": int(seed),
        "only_variable": "model.head temporal decoder",
        "base_config": str(base_path),
        "base_config_sha256": _sha256(base_path),
        "stack_input_mask": list(STACK_MASK),
        "parameter_budget_tolerance": 0.05,
    }
    return config


def parameter_counts(base: dict, arms: Iterable[str]) -> Dict[str, int]:
    """Instantiate heads without loading the large backbone."""

    from src.models.heads import (
        CausalAttentionCountHead,
        CausalGRUCountHead,
        CausalSSMCountHead,
        DeformableCountHead,
        TCNCountHead,
    )

    stack_dims = _parse_int_list(base["model"]["encoder"]["encoder_dim"])
    d_in = sum(stack_dims)
    classes = int(base["model"].get("num_count_classes", 4))
    constructors = {
        "tcn": lambda h: TCNCountHead(
            d_in=d_in, num_classes=classes, stack_dims=stack_dims,
            d_model=h["d_model"], dilations=tuple(h["dilations"]),
            kernel=h["kernel"], dropout=h["dropout"],
            stack_input_mask=STACK_MASK),
        "deformable": lambda h: DeformableCountHead(
            d_in=d_in, num_classes=classes, stack_dims=stack_dims,
            d_model=h["d_model"], num_layers=h["num_layers"],
            num_points=h["num_points"], num_groups=h["num_groups"],
            max_offset=h["max_offset"], dropout=h["dropout"],
            stack_input_mask=STACK_MASK),
        "gru": lambda h: CausalGRUCountHead(
            d_in=d_in, num_classes=classes, stack_dims=stack_dims,
            d_model=h["d_model"], num_layers=h["num_layers"],
            dropout=h["dropout"], stack_input_mask=STACK_MASK),
        "ssm": lambda h: CausalSSMCountHead(
            d_in=d_in, num_classes=classes, stack_dims=stack_dims,
            d_model=h["d_model"], state_dim=h["state_dim"],
            num_layers=h["num_layers"], dropout=h["dropout"],
            stack_input_mask=STACK_MASK),
        "attention": lambda h: CausalAttentionCountHead(
            d_in=d_in, num_classes=classes, stack_dims=stack_dims,
            d_model=h["d_model"], num_heads=h["num_heads"],
            num_layers=h["num_layers"], max_context=h["max_context"],
            ffn_multiplier=h["ffn_multiplier"], dropout=h["dropout"],
            stack_input_mask=STACK_MASK),
    }
    counts = {}
    for arm in arms:
        model = constructors[arm](ARM_HEADS[arm])
        counts[arm] = sum(parameter.numel() for parameter in model.parameters())
    return counts


def assert_matched_budget(counts: Dict[str, int], tolerance: float = 0.05) -> None:
    reference = counts["tcn"]
    outside = {
        arm: count
        for arm, count in counts.items()
        if abs(count / reference - 1.0) > tolerance
    }
    if outside:
        raise RuntimeError(
            f"head parameter budget drifted outside ±{tolerance:.0%} of "
            f"TCN={reference:,}: {outside}")


def _csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--init-from", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arms", default=",".join(ARM_HEADS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--execute", action="store_true",
        help="Run training after preparing configs (default: prepare only)")
    args = parser.parse_args()

    base_path = args.base_config.resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"base config not found: {base_path}")
    with base_path.open() as stream:
        base = yaml.safe_load(stream)

    arms = _csv(args.arms)
    unknown = sorted(set(arms) - set(ARM_HEADS))
    if unknown or not arms:
        raise ValueError(f"invalid arms {unknown}; available: {sorted(ARM_HEADS)}")
    seeds = [int(seed) for seed in _csv(args.seeds)]
    if not seeds:
        raise ValueError("at least one seed is required")

    counts = parameter_counts(base, arms)
    # The TCN reference is needed even when the caller requests a subset.
    if "tcn" not in counts:
        counts.update(parameter_counts(base, ["tcn"]))
    assert_matched_budget(counts)
    print("Parameter-matched heads:")
    for arm in arms:
        print(f"  {arm:12s} {counts[arm]:,}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    for arm in arms:
        for seed in seeds:
            config = make_config(base, arm, seed, base_path)
            if args.max_steps > 0:
                config["training"]["max_steps"] = args.max_steps
            path = output_dir / f"{arm}_s{seed}.yaml"
            with path.open("w") as stream:
                yaml.safe_dump(config, stream, sort_keys=False)
            prepared.append((arm, seed, path, config["training"]["log_dir"]))

    init_path = args.init_from.resolve()
    print(f"Prepared {len(prepared)} configs in {output_dir}")
    for _, _, path, log_dir in prepared:
        command = [
            sys.executable, "src/train.py", "--config", str(path),
            "--init-from", str(init_path), "--log-dir", log_dir,
        ]
        print(" ".join(command))

    if not args.execute:
        return
    if not init_path.is_file():
        raise FileNotFoundError(
            f"backbone initialization checkpoint not found: {init_path}")
    train_manifest = Path(base["data"]["train_manifest"]).expanduser()
    if not train_manifest.is_file():
        raise FileNotFoundError(
            "the locked historical training manifest is unavailable: "
            f"{train_manifest}. Restore that manifest/data or pass a base "
            "config for a new explicitly named protocol; do not silently "
            "substitute a different data mixture in this bake-off.")
    try:
        __import__("k2")
    except ImportError as error:
        raise RuntimeError(
            "the active Python environment has no k2; run with the project's "
            ".zipformer environment") from error
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    for _, _, path, log_dir in prepared:
        command = [
            sys.executable, "src/train.py", "--config", str(path),
            "--init-from", str(init_path), "--log-dir", log_dir,
        ]
        subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
