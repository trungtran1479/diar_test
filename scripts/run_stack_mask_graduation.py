#!/usr/bin/env python3
"""Matched three-seed graduation of sparse Zipformer stack *views*.

This is deliberately independent of the Phase-8 chain.  It never mutates the
chain base or decision log.  Training requires the explicit ``run`` command;
``preflight`` and ``report`` never launch a training process.

See ``STACK_MASK_GRADUATION_PREREG.md`` for the locked protocol and decision
rules.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, NoReturn, Tuple

import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[1]
PY = Path("/home/edabk/miniconda3/envs/.zipformer/bin/python")
BASE_CONFIG = REPO / "configs/zipcount_v4_distill.yaml"
INIT = REPO / "logs/r8_v3_diverse/best_macro_f1.pt"
TRAIN_MANIFEST = Path(
    "/media/edabk/500GB Hard Disk/data_diar/manifests/train_manifest_v3.json"
)
VAL_MANIFEST = REPO / "data/meetings/val_manifest.json"
VOX_SEL = Path(
    "/media/edabk/500GB Hard Disk/data_diar/voxconverse/vox_sel.json"
)
PREREG = REPO / "STACK_MASK_GRADUATION_PREREG.md"
ART = REPO / "artifacts/stack_mask_graduation"
RESULTS = REPO / "results"
LOGS = REPO / "logs"

SEEDS = (1234, 2345, 3456)
STEPS = 3000
NONINF_MARGIN = -0.002
N_BOOT = 20_000
BOOT_SEED = 0
NEED_DISK_GB = 16

ARMS: Mapping[str, Tuple[int, ...]] = {
    "all6": (1, 1, 1, 1, 1, 1),
    "only1": (0, 1, 0, 0, 0, 0),
    "pre012": (1, 1, 1, 0, 0, 0),
}
RUN_ORDER = (
    (1234, "all6"), (1234, "only1"), (1234, "pre012"),
    (2345, "only1"), (2345, "pre012"), (2345, "all6"),
    (3456, "pre012"), (3456, "all6"), (3456, "only1"),
)
STACK_DIMS = (192, 256, 384, 512, 384, 256)

ZIPFORMER_DIR = "third_party/icefall/egs/librispeech/ASR/zipformer"
SOURCE_FILES = (
    "scripts/run_stack_mask_graduation.py",
    "STACK_MASK_GRADUATION_PREREG.md",
    "configs/zipcount_v4_distill.yaml",
    "src/train.py",
    "src/models/zipcount_v1.py",
    "src/models/zipformer_wrapper.py",
    "src/models/heads.py",
    "src/models/deformable.py",
    "src/models/pyramid_head.py",
    "src/models/losses.py",
    "src/models/structured_losses.py",
    "src/models/crnn_baseline.py",
    "src/data/dataset.py",
    "src/data/label_utils.py",
    "src/data/feature_extractor.py",
    "src/utils/metrics.py",
    "scripts/eval_per_recording.py",
    f"{ZIPFORMER_DIR}/zipformer.py",
    f"{ZIPFORMER_DIR}/scaling.py",
    f"{ZIPFORMER_DIR}/subsampling.py",
    f"{ZIPFORMER_DIR}/model.py",
    f"{ZIPFORMER_DIR}/finetune.py",
    # finetune.py imports these modules while constructing the full RNNT
    # model.  Decoder/joiner parameters are discarded afterwards, but their
    # constructors consume RNG before the count head is initialized; they are
    # therefore part of the matched-initialization protocol.
    f"{ZIPFORMER_DIR}/encoder_interface.py",
    f"{ZIPFORMER_DIR}/decoder.py",
    f"{ZIPFORMER_DIR}/joiner.py",
    f"{ZIPFORMER_DIR}/optim.py",
    f"{ZIPFORMER_DIR}/asr_datamodule.py",
)


def fail(message: str) -> NoReturn:
    raise SystemExit(f"FATAL: {message}")


def sha_file(path: Path, chunk_size: int = 1 << 22) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"missing or empty file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_yaml(config: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(config), sort_keys=True)


def config_sha(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_yaml(config).encode()).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def source_snapshot() -> Dict[str, Any]:
    files = {rel: sha_file(REPO / rel) for rel in SOURCE_FILES}
    return {
        "files": files,
        "combined_sha": hashlib.sha256(canonical_json(files)).hexdigest(),
    }


def assert_source_snapshot(expected: Mapping[str, Any], where: str) -> None:
    live = source_snapshot()
    if live != expected:
        changed = sorted(
            rel for rel in set(expected.get("files", {})) | set(live["files"])
            if expected.get("files", {}).get(rel) != live["files"].get(rel)
        )
        fail(
            f"source freeze violated {where}; changed={changed}. "
            "The in-flight run is uncertified."
        )


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    with tmp.open("x") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def tag(arm: str, seed: int) -> str:
    if arm not in ARMS or seed not in SEEDS:
        fail(f"invalid arm/seed: {arm}/{seed}")
    return f"stackgrad_{arm}_s{seed}"


def paths_for(arm: str, seed: int) -> Dict[str, Path]:
    name = tag(arm, seed)
    return {
        "config": ART / f"{name}.yaml",
        "sidecar": ART / f"{name}.lineage.json",
        "result": RESULTS / f"{name}_voxsel.json",
        "checkpoint": LOGS / name / f"step{STEPS}.pt",
        "log_dir": LOGS / name,
        "log": LOGS / f"{name}.log",
    }


def make_config(arm: str, seed: int) -> Dict[str, Any]:
    if arm not in ARMS:
        fail(f"unknown arm: {arm}")
    with BASE_CONFIG.open() as stream:
        config = yaml.safe_load(stream)

    encoder = config["model"]["encoder"]
    encoder["multiscale"] = True
    encoder["multiscale_alignment"] = "average"
    encoder["multiscale_include_final"] = False
    config["model"]["head"] = {
        "type": "tcn_ordinal",
        "d_model": 192,
        "dropout": 0.1,
        "stack_input_mask": list(ARMS[arm]),
    }

    # Verbatim legacy-v4 objectives; only later experimental losses and KD
    # are disabled.  This is the clean P8A-average training regime.
    loss = config["loss"]
    loss["type"] = "legacy"
    loss["lambda_boundary_smooth"] = 0.0
    loss["lambda_event"] = 0.0
    loss["lambda_raw_anchor"] = 0.0
    config["distill"]["teacher_dir"] = ""
    config["distill"]["lambda_kd"] = 0.0

    name = tag(arm, seed)
    training = config["training"]
    training.update({
        "stage": "stage1_freeze_backbone",
        "freeze_backbone_all": True,
        "init_backbone_only": True,
        "require_backbone_complete": True,
        "reset_rng_after_init": True,
        "batch_size": 24,
        "lr": 1.0e-3,
        "lr_schedule": "cosine",
        "warmup_steps": 500,
        "max_steps": STEPS,
        "grad_clip": 5.0,
        "use_amp": True,
        "amp_init_scale": 4096.0,
        "amp_growth_interval": 2000,
        "max_amp_overflows": 0,
        "num_workers": 12,
        "seed": seed,
        "eval_interval": 300,
        "eval_interval_early": 300,
        "early_phase_steps": STEPS,
        "save_every_eval": True,
        "snapshot_weights_only": True,
        "save_best": False,
        "save_last": False,
        "log_dir": f"logs/{name}",
    })
    training.pop("backbone_lr", None)
    training.pop("head_lr", None)

    config["data"]["train_manifest"] = str(TRAIN_MANIFEST)
    config["data"]["val_manifest"] = str(VAL_MANIFEST)
    return config


def protocol_inputs(
    config: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> Dict[str, Any]:
    data = config["data"]
    encoder = config["model"]["encoder"]
    stats_path = Path(data["stats_file"])
    encoder_checkpoint = Path(encoder["checkpoint"])
    return {
        "config_sha": config_sha(config),
        "init_sha": sha_file(INIT),
        "train_manifest_sha": sha_file(Path(data["train_manifest"])),
        "val_manifest_sha": sha_file(Path(data["val_manifest"])),
        "eval_manifest_sha": sha_file(VOX_SEL),
        "stats_sha": sha_file(stats_path),
        "encoder_pretrained_sha": sha_file(encoder_checkpoint),
        "source_combined_sha": snapshot["combined_sha"],
        "source_files": snapshot["files"],
        "locked_step": STEPS,
    }


def output_fingerprint(arm: str, seed: int) -> Dict[str, Any]:
    p = paths_for(arm, seed)
    with p["config"].open() as stream:
        disk_config = yaml.safe_load(stream)
    with p["result"].open() as stream:
        result = json.load(stream)
    if result.get("manifest") != str(VOX_SEL):
        fail(f"{tag(arm, seed)} result names the wrong evaluation manifest")
    if int(result.get("n_recordings", -1)) <= 0:
        fail(f"{tag(arm, seed)} result contains no recordings")
    if result.get("checkpoint") not in {
        str(p["checkpoint"].relative_to(REPO)), str(p["checkpoint"])
    }:
        fail(f"{tag(arm, seed)} result names the wrong checkpoint")
    return {
        "config_sha": config_sha(disk_config),
        "config_file_sha": sha_file(p["config"]),
        "checkpoint_sha": sha_file(p["checkpoint"]),
        "result_sha": sha_file(p["result"]),
        "n_recordings": int(result["n_recordings"]),
    }


def lineage_valid(
    arm: str, seed: int, wanted_yaml: str, wanted_inputs: Mapping[str, Any]
) -> bool:
    p = paths_for(arm, seed)
    required = ("config", "sidecar", "result", "checkpoint")
    if not all(p[key].is_file() and p[key].stat().st_size > 0
               for key in required):
        return False
    try:
        if p["config"].read_text() != wanted_yaml:
            return False
        with p["sidecar"].open() as stream:
            stored = json.load(stream)
        if stored.get("inputs") != wanted_inputs:
            return False
        return stored.get("outputs") == output_fingerprint(arm, seed)
    except (
        SystemExit, OSError, ValueError, KeyError, TypeError,
        json.JSONDecodeError,
    ):
        return False


def unique_stale_path(path: Path, stamp: str) -> Path:
    candidate = path.with_name(f"{path.name}.stale.{stamp}")
    serial = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.stale.{stamp}.{serial}")
        serial += 1
    return candidate


def invalidate(arm: str, seed: int, reason: str) -> None:
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    p = paths_for(arm, seed)
    moved = []
    for key in ("result", "config", "sidecar", "log", "log_dir"):
        path = p[key]
        if path.exists():
            destination = unique_stale_path(path, stamp)
            path.rename(destination)
            moved.append(str(destination.relative_to(REPO)))
    if moved:
        print(f"[stale] {tag(arm, seed)}: {reason}: {moved}", flush=True)


def tensor_digest(named_tensors: Iterable[Tuple[str, Any]]) -> str:
    """Hash tensor names, schemas, and exact CPU bytes deterministically."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(canonical_json(list(value.shape)))
        # Flatten first: PyTorch rejects a dtype-changing view directly on a
        # zero-dimensional tensor even though its storage has valid bytes.
        # The flat byte view also handles empty tensors deterministically.
        digest.update(
            value.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def state_schema(model: Any) -> Tuple[Tuple[str, str, Tuple[int, ...]], ...]:
    return tuple(
        (name, str(value.dtype), tuple(value.shape))
        for name, value in model.state_dict().items()
    )


def parameter_schema(
    model: Any,
) -> Tuple[Tuple[str, str, Tuple[int, ...], bool], ...]:
    return tuple(
        (name, str(value.dtype), tuple(value.shape), bool(value.requires_grad))
        for name, value in model.named_parameters()
    )


def reset_initialization_rng(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_exact_v3_backbone(model: Any) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    import torch

    checkpoint = torch.load(INIT, map_location="cpu", weights_only=False)
    full_state = checkpoint.get("model_state_dict", checkpoint)
    backbone_state = {
        name: value for name, value in full_state.items()
        if name.startswith("backbone.")
    }
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    backbone_missing = [name for name in missing if name.startswith("backbone.")]
    if backbone_missing:
        fail(
            "preflight require_backbone_complete failed; first missing keys: "
            f"{backbone_missing[:5]}"
        )
    if unexpected:
        fail(f"v3 backbone produced unexpected keys: {unexpected[:5]}")
    return tuple(sorted(missing)), tuple(sorted(unexpected))


def perturbation_output(head: Any, value: Any) -> Any:
    output = head(value)
    return output[0] if isinstance(output, (tuple, list)) else output


def check_mask_semantics(models: Mapping[str, Any], seed: int) -> None:
    import torch

    offsets = np.cumsum((0,) + STACK_DIMS)
    generator = torch.Generator().manual_seed(seed + 90_000)
    base = torch.randn(2, 11, sum(STACK_DIMS), generator=generator)
    perturb = torch.randn(2, 11, sum(STACK_DIMS), generator=generator)
    for arm, model in models.items():
        head = model.head.eval()
        mask = ARMS[arm]
        with torch.no_grad():
            reference = perturbation_output(head, base)
            included = base.clone()
            excluded = base.clone()
            for index, enabled in enumerate(mask):
                start, end = int(offsets[index]), int(offsets[index + 1])
                if enabled:
                    included[..., start:end] += perturb[..., start:end]
                else:
                    excluded[..., start:end] += perturb[..., start:end]
            included_delta = (
                perturbation_output(head, included) - reference
            ).abs().max().item()
            excluded_delta = (
                perturbation_output(head, excluded) - reference
            ).abs().max().item()
        if included_delta <= 1e-7:
            fail(f"{arm}/seed{seed}: included stacks do not affect output")
        if 0 in mask and excluded_delta > 1e-7:
            fail(
                f"{arm}/seed{seed}: excluded stacks leak into output "
                f"(max delta {excluded_delta:.3e})"
            )


def model_equivalence_preflight() -> Dict[str, Any]:
    """Prove all arms have the same graph and exact post-init tensor values."""
    import torch

    sys.path.insert(0, str(REPO))
    from src.models.zipcount_v1 import build_model

    summary: Dict[str, Any] = {}
    for seed in SEEDS:
        models: Dict[str, Any] = {}
        rows: Dict[str, Any] = {}
        for arm in ARMS:
            config = make_config(arm, seed)
            reset_initialization_rng(seed)
            model = build_model(copy.deepcopy(config)).cpu()
            missing, unexpected = load_exact_v3_backbone(model)
            for parameter in model.backbone.parameters():
                parameter.requires_grad = False
            models[arm] = model
            rows[arm] = {
                "state_schema": state_schema(model),
                "parameter_schema": parameter_schema(model),
                "state_sha": tensor_digest(model.state_dict().items()),
                "parameter_init_sha": tensor_digest(model.named_parameters()),
                "head_init_sha": tensor_digest(model.head.named_parameters()),
                "total_params": sum(p.numel() for p in model.parameters()),
                "trainable_params": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "missing_after_v3_backbone": missing,
                "unexpected_after_v3_backbone": unexpected,
            }
        reference = rows["all6"]
        invariant_keys = (
            "state_schema", "parameter_schema", "state_sha",
            "parameter_init_sha", "head_init_sha", "total_params",
            "trainable_params", "missing_after_v3_backbone",
            "unexpected_after_v3_backbone",
        )
        for arm, row in rows.items():
            for key in invariant_keys:
                if row[key] != reference[key]:
                    fail(
                        f"matched-init preflight failed for seed {seed}, "
                        f"{arm}: {key} differs from all6"
                    )
        check_mask_semantics(models, seed)
        summary[str(seed)] = {
            key: reference[key]
            for key in (
                "state_sha", "parameter_init_sha", "head_init_sha",
                "total_params", "trainable_params",
                "missing_after_v3_backbone",
                "unexpected_after_v3_backbone",
            )
        }
        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summary


def validate_static_protocol(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    if not PY.is_file():
        fail(f"Python environment not found: {PY}")
    for path in (BASE_CONFIG, INIT, TRAIN_MANIFEST, VAL_MANIFEST, VOX_SEL, PREREG):
        sha_file(path)
    if tuple((seed, arm) for seed, arm in RUN_ORDER) != (
        (1234, "all6"), (1234, "only1"), (1234, "pre012"),
        (2345, "only1"), (2345, "pre012"), (2345, "all6"),
        (3456, "pre012"), (3456, "all6"), (3456, "only1"),
    ):
        fail("locked Latin-square run order changed")
    for seed in SEEDS:
        configs = {arm: make_config(arm, seed) for arm in ARMS}
        reference = copy.deepcopy(configs["all6"])
        reference["model"]["head"].pop("stack_input_mask")
        reference["training"].pop("log_dir")
        for arm, config in configs.items():
            candidate = copy.deepcopy(config)
            mask = tuple(candidate["model"]["head"].pop("stack_input_mask"))
            candidate["training"].pop("log_dir")
            if mask != ARMS[arm]:
                fail(f"{arm}: wrong stack_input_mask {mask}")
            if candidate != reference:
                fail(f"{arm}/seed{seed}: config differs beyond stack_input_mask")
    assert_source_snapshot(snapshot, "during static preflight")
    summary = model_equivalence_preflight()
    assert_source_snapshot(snapshot, "after matched-model preflight")
    return summary


def disk_preflight() -> None:
    usage = shutil.disk_usage(REPO)
    free_gb = usage.free / 1024 ** 3
    if free_gb < NEED_DISK_GB:
        fail(
            f"need {NEED_DISK_GB} GiB free on repository filesystem, "
            f"found {free_gb:.1f} GiB"
        )


def gpu_preflight() -> None:
    code = (
        "import torch\n"
        "assert torch.cuda.is_available(), 'CUDA unavailable'\n"
        "free,total=torch.cuda.mem_get_info()\n"
        "print(f'GPU={torch.cuda.get_device_name(0)} "
        "free={free/1024**3:.1f}/{total/1024**3:.1f}GiB')\n"
        "assert free >= 20*1024**3, 'less than 20 GiB GPU memory free'\n"
    )
    subprocess.run([str(PY), "-c", code], cwd=REPO, check=True)
    active = subprocess.run(
        ["pgrep", "-af", r"src/train\.py"], text=True,
        stdout=subprocess.PIPE, check=False,
    ).stdout.strip()
    if active:
        fail(f"another training process is active:\n{active}")


def current_protocol_record(
    snapshot: Mapping[str, Any], preflight_summary: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "protocol": "stack-mask-graduation-v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_snapshot": snapshot,
        "preflight": preflight_summary,
        "seeds": list(SEEDS),
        "arms": {arm: list(mask) for arm, mask in ARMS.items()},
        "steps": STEPS,
        "run_order": [list(item) for item in RUN_ORDER],
    }


def install_protocol_record(record: Mapping[str, Any]) -> None:
    path = ART / "protocol_snapshot.json"
    if path.exists():
        try:
            with path.open() as stream:
                old = json.load(stream)
        except (OSError, json.JSONDecodeError):
            old = None
        # Normalize tuples and any other JSON-compatible containers exactly as
        # they appear after persistence; otherwise every resume would stale a
        # byte-identical protocol merely because JSON decoded tuples as lists.
        comparable_old = copy.deepcopy(old)
        comparable_new = json.loads(canonical_json(record))
        if isinstance(comparable_old, dict):
            comparable_old.pop("created_at", None)
        comparable_new.pop("created_at", None)
        if comparable_old == comparable_new:
            return
        destination = unique_stale_path(
            path, dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        )
        path.rename(destination)
        print(f"[stale] protocol snapshot -> {destination.relative_to(REPO)}")
    atomic_json(path, record)


def arm_has_any_artifact(arm: str, seed: int) -> bool:
    return any(path.exists() for path in paths_for(arm, seed).values())


def remove_intermediate_checkpoints(log_dir: Path) -> None:
    locked = log_dir / f"step{STEPS}.pt"
    if not locked.is_file() or locked.stat().st_size == 0:
        fail(f"locked checkpoint absent: {locked}")
    for path in log_dir.glob("step*.pt"):
        if path != locked:
            path.unlink()


def run_one(
    arm: str, seed: int, snapshot: Mapping[str, Any]
) -> Path:
    name = tag(arm, seed)
    p = paths_for(arm, seed)
    config = make_config(arm, seed)
    wanted_yaml = canonical_yaml(config)
    wanted_inputs = protocol_inputs(config, snapshot)

    if lineage_valid(arm, seed, wanted_yaml, wanted_inputs):
        print(f"[skip] {name}: input/output lineage verified", flush=True)
        return p["result"]
    if arm_has_any_artifact(arm, seed):
        invalidate(arm, seed, "incomplete or lineage drift")

    assert_source_snapshot(snapshot, f"before {name}")
    disk_preflight()
    atomic_text(p["config"], wanted_yaml)
    # Bind the object used to configure training to the exact on-disk bytes.
    with p["config"].open() as stream:
        disk_config = yaml.safe_load(stream)
    if disk_config != config or config_sha(disk_config) != wanted_inputs["config_sha"]:
        fail(f"{name}: on-disk config differs from the preflighted config")

    print(
        f"[run ] {name} config={wanted_inputs['config_sha'][:12]} "
        f"source={wanted_inputs['source_combined_sha'][:12]}",
        flush=True,
    )
    start = time.time()
    with p["log"].open("w") as log:
        subprocess.run(
            [
                str(PY), "src/train.py", "--config", str(p["config"]),
                "--init-from", str(INIT),
            ],
            cwd=REPO,
            env={**os.environ, "PYTHONPATH": "."},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if not p["checkpoint"].is_file() or p["checkpoint"].stat().st_size == 0:
        fail(f"{name}: training finished without locked step-{STEPS} checkpoint")

    subprocess.run(
        [
            str(PY), "scripts/eval_per_recording.py",
            "--manifest", str(VOX_SEL),
            "--checkpoint", str(p["checkpoint"].relative_to(REPO)),
            "--config", str(p["config"].relative_to(REPO)),
            "--out", str(p["result"].relative_to(REPO)),
        ],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": "."},
        check=True,
    )

    # Re-read all protocol inputs after both train and evaluation.  This also
    # re-hashes the config object against the file before certification.
    assert_source_snapshot(snapshot, f"after {name}")
    with p["config"].open() as stream:
        disk_config = yaml.safe_load(stream)
    inputs_after = protocol_inputs(disk_config, snapshot)
    if inputs_after != wanted_inputs or p["config"].read_text() != wanted_yaml:
        fail(f"{name}: protocol/config changed during the run; uncertified")

    outputs = output_fingerprint(arm, seed)
    if outputs["config_sha"] != wanted_inputs["config_sha"]:
        fail(f"{name}: output config is not the input config")
    atomic_json(
        p["sidecar"],
        {
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "inputs": wanted_inputs,
            "outputs": outputs,
        },
    )
    # The completion sidecar now certifies the locked output, so deleting only
    # this runner's intermediate snapshots is safe and bounds disk use.
    remove_intermediate_checkpoints(p["log_dir"])
    print(f"[done] {name} in {(time.time() - start) / 60:.1f} min", flush=True)
    return p["result"]


def result_paths(arm: str) -> List[Path]:
    return [paths_for(arm, seed)["result"] for seed in SEEDS]


def load_results(arm: str) -> List[Dict[str, Any]]:
    rows = []
    for seed, path in zip(SEEDS, result_paths(arm)):
        if not path.is_file() or path.stat().st_size == 0:
            fail(f"missing result for report: {path}")
        sidecar = paths_for(arm, seed)["sidecar"]
        if not sidecar.is_file():
            fail(f"missing completion sidecar for report: {sidecar}")
        with path.open() as stream:
            rows.append(json.load(stream))
    return rows


def compare(control: str, treatment: str) -> Dict[str, Any]:
    a_runs = load_results(control)
    b_runs = load_results(treatment)
    all_runs = a_runs + b_runs
    recording_sets = [frozenset(row["per_recording"]) for row in all_runs]
    if len(set(recording_sets)) != 1:
        fail(
            f"{treatment} vs {control}: recording sets differ; paired "
            "intersection is forbidden"
        )
    recordings = sorted(recording_sets[0])
    if not recordings:
        fail(f"{treatment} vs {control}: no recordings")

    per_seed = [
        float(b["pooled"]["macro_f1"] - a["pooled"]["macro_f1"])
        for a, b in zip(a_runs, b_runs)
    ]
    delta_f1_2 = float(np.mean([
        b["pooled"]["f1_2"] - a["pooled"]["f1_2"]
        for a, b in zip(a_runs, b_runs)
    ]))
    delta_osd = float(np.mean([
        b["pooled"]["osd_f1"] - a["pooled"]["osd_f1"]
        for a, b in zip(a_runs, b_runs)
    ]))
    rec_a = np.asarray([
        [row["per_recording"][recording]["macro_f1"]
         for recording in recordings]
        for row in a_runs
    ]).mean(axis=0)
    rec_b = np.asarray([
        [row["per_recording"][recording]["macro_f1"]
         for recording in recordings]
        for row in b_runs
    ]).mean(axis=0)
    per_recording_delta = rec_b - rec_a
    rng = np.random.default_rng(BOOT_SEED)
    boot = per_recording_delta[
        rng.integers(
            0, len(recordings), size=(N_BOOT, len(recordings))
        )
    ].mean(axis=1)
    ci = [
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    ]
    mean_pooled = float(np.mean(per_seed))
    practical_r2 = mean_pooled >= 0.005
    practical_r3 = (
        (delta_f1_2 >= 0.010 or delta_osd >= 0.010)
        and mean_pooled >= NONINF_MARGIN
    )
    operational_ni = mean_pooled >= NONINF_MARGIN
    strict_ni = (
        all(delta >= NONINF_MARGIN for delta in per_seed)
        and ci[0] > NONINF_MARGIN
    )
    if all(delta > 0 for delta in per_seed) and ci[0] > 0:
        evidence = "REAL"
    elif all(delta < 0 for delta in per_seed) and ci[1] < 0:
        evidence = "NEGATIVE"
    elif mean_pooled > 0:
        evidence = "SUGGESTIVE"
    else:
        evidence = "NONE"
    return {
        "control": control,
        "treatment": treatment,
        "per_seed": dict(zip(map(str, SEEDS), per_seed)),
        "mean_pooled_delta": mean_pooled,
        "delta_f1_2": delta_f1_2,
        "delta_osd": delta_osd,
        "bootstrap": {
            "unit": "recording",
            "seeds_averaged_within_recording": True,
            "n_boot": N_BOOT,
            "rng_seed": BOOT_SEED,
            "mean_delta": float(per_recording_delta.mean()),
            "ci95": ci,
            "n_improved": int((per_recording_delta > 0).sum()),
            "n_recordings": len(recordings),
        },
        "evidence": evidence,
        "operational_mean_only_noninferior": operational_ni,
        "strict_noninferior": strict_ni,
        "practical_r2": practical_r2,
        "practical_r3": practical_r3,
        "practical_adoption": practical_r2 or practical_r3,
    }


def decide(comparisons: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    only_ni = bool(comparisons["only1_vs_all6"]["strict_noninferior"])
    pre_ni = bool(comparisons["pre012_vs_all6"]["strict_noninferior"])
    if not only_ni and not pre_ni:
        slim = "none"
        reason = "neither sparse view passes strict non-inferiority"
    elif only_ni and not pre_ni:
        slim = "only1"
        reason = "only1 alone passes strict non-inferiority"
    elif pre_ni and not only_ni:
        slim = "pre012"
        reason = "pre012 alone passes strict non-inferiority"
    else:
        direct = comparisons["pre012_vs_only1"]
        if direct["practical_adoption"]:
            slim = "pre012"
            reason = "both strict-NI; pre012 beats only1 through R2/R3"
        else:
            slim = "only1"
            reason = "both strict-NI; fixed simplicity preference keeps only1"

    practical = [
        arm for arm in ("only1", "pre012")
        if comparisons[f"{arm}_vs_all6"]["practical_adoption"]
    ]
    if not practical:
        mask_adoption = "all6"
        adoption_reason = "no sparse view passes direct R2/R3 vs all6"
    elif practical == ["only1"]:
        mask_adoption = "only1"
        adoption_reason = "only1 passes direct R2/R3 vs all6"
    elif practical == ["pre012"]:
        mask_adoption = "pre012"
        adoption_reason = "pre012 passes direct R2/R3 vs all6"
    else:
        direct = comparisons["pre012_vs_only1"]
        if direct["practical_adoption"]:
            mask_adoption = "pre012"
            adoption_reason = "both pass vs all6; pre012 also passes R2/R3 vs only1"
        else:
            mask_adoption = "only1"
            adoption_reason = "both pass vs all6; simplicity tie-break keeps only1"
    return {
        "slim_followup_authorized": slim != "none",
        "slim_followup_candidate": slim,
        "slim_reason": reason,
        "current_mask_adoption_recommendation": mask_adoption,
        "adoption_reason": adoption_reason,
        "phase8_base_mutated": False,
    }


def certified_result_lineage(
    snapshot: Mapping[str, Any],
) -> Dict[str, Any]:
    lineage: Dict[str, Any] = {}
    for arm in ARMS:
        lineage[arm] = {}
        for seed in SEEDS:
            p = paths_for(arm, seed)
            config = make_config(arm, seed)
            wanted_yaml = canonical_yaml(config)
            wanted_inputs = protocol_inputs(config, snapshot)
            if not lineage_valid(
                arm, seed, wanted_yaml, wanted_inputs
            ):
                fail(
                    f"report refuses stale/uncertified arm: "
                    f"{tag(arm, seed)}"
                )
            with p["sidecar"].open() as stream:
                sidecar = json.load(stream)
            live = output_fingerprint(arm, seed)
            if sidecar.get("outputs") != live:
                fail(f"report lineage drift: {tag(arm, seed)}")
            lineage[arm][str(seed)] = {
                "result": str(p["result"].relative_to(REPO)),
                "sidecar_sha": sha_file(p["sidecar"]),
                "outputs": live,
            }
    return lineage


def build_report() -> Dict[str, Any]:
    snapshot = source_snapshot()
    # Authenticate every input and output before reading metric values.  A
    # tampered JSON must never influence even an ultimately rejected report.
    lineage = certified_result_lineage(snapshot)
    comparisons = {
        "only1_vs_all6": compare("all6", "only1"),
        "pre012_vs_all6": compare("all6", "pre012"),
        "pre012_vs_only1": compare("only1", "pre012"),
    }
    return {
        "protocol": "stack-mask-graduation-v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rules": {
            "noninferiority_margin": NONINF_MARGIN,
            "strict_noninferiority":
                "all seed deltas >= margin and bootstrap lower CI > margin",
            "r2": "mean pooled delta >= +0.005",
            "r3":
                "delta class-2 or OSD >= +0.010 and macro delta >= -0.002",
        },
        "comparisons": comparisons,
        "decision": decide(comparisons),
        "source_snapshot": snapshot,
        "lineage": lineage,
    }


def print_report(report: Mapping[str, Any]) -> None:
    print("\n=== STACK MASK GRADUATION: LOCKED STEP 3000 ===")
    for key in ("only1_vs_all6", "pre012_vs_all6", "pre012_vs_only1"):
        row = report["comparisons"][key]
        ci = row["bootstrap"]["ci95"]
        deltas = "/".join(f"{x:+.4f}" for x in row["per_seed"].values())
        print(
            f"{key:<20} seeds={deltas} mean={row['mean_pooled_delta']:+.4f} "
            f"bootCI=[{ci[0]:+.4f},{ci[1]:+.4f}] "
            f"opNI={row['operational_mean_only_noninferior']} "
            f"strictNI={row['strict_noninferior']} "
            f"adopt={row['practical_adoption']} evidence={row['evidence']}"
        )
    decision = report["decision"]
    print(
        f"slim follow-up: {decision['slim_followup_candidate']} "
        f"({decision['slim_reason']})"
    )
    print(
        "current-mask recommendation: "
        f"{decision['current_mask_adoption_recommendation']} "
        f"({decision['adoption_reason']})"
    )
    print("Phase-8 base was not modified.")


def acquire_lock() -> Any:
    ART.mkdir(parents=True, exist_ok=True)
    lock_path = ART / "runner.lock"
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fail(f"another graduation runner holds {lock_path}")
    lock.seek(0)
    lock.truncate()
    lock.write(f"pid={os.getpid()} started={dt.datetime.now().isoformat()}\n")
    lock.flush()
    return lock


def preflight(require_runtime_capacity: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    snapshot = source_snapshot()
    disk_preflight()
    if require_runtime_capacity:
        gpu_preflight()
    summary = validate_static_protocol(snapshot)
    print("Matched-model preflight passed:")
    for seed, row in summary.items():
        print(
            f"  seed {seed}: total={row['total_params']:,} "
            f"trainable={row['trainable_params']:,} "
            f"state={row['state_sha'][:12]} head={row['head_init_sha'][:12]}"
        )
    return snapshot, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "run", "report"),
        help="training occurs only for the explicit 'run' command",
    )
    args = parser.parse_args()

    if args.command == "preflight":
        preflight(require_runtime_capacity=False)
        print("Preflight passed; no training or result artifact was created.")
        return

    lock = acquire_lock()
    # Keep a live reference for the process lifetime; flock is released on exit.
    _process_lock = lock
    if args.command == "run":
        snapshot, summary = preflight(require_runtime_capacity=True)
        ART.mkdir(parents=True, exist_ok=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        LOGS.mkdir(parents=True, exist_ok=True)
        install_protocol_record(current_protocol_record(snapshot, summary))
        for seed, arm in RUN_ORDER:
            run_one(arm, seed, snapshot)
        assert_source_snapshot(snapshot, "after all arms")
        report = build_report()
        atomic_json(ART / "final_report.json", report)
        print_report(report)
    else:
        report = build_report()
        atomic_json(ART / "final_report.json", report)
        print_report(report)
    assert _process_lock is not None


if __name__ == "__main__":
    main()
