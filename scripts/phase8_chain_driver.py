"""Phase 8 chain, revision 8 — legacy-bridge fallback after phase b failed.

Revision-8: phase b's structured (SORD/CORN) bridge FAILED both ways
(sord -0.0087, cumulative -0.0122 vs the a0 carried config; see
decisions.jsonl), which under the OLD design stopped the chain entirely --
b2/c/c2/d1/d2/e are all increments on `structured_loss_block` and cannot run
meaningfully on a carried_cfg still using `loss.type: legacy`. That is a real
structural block, not researcher fatigue: mut_b2 in particular is a landmine
under legacy loss (CORN mode makes the model emit count_logits already in
LOG-PROBABILITY space; the legacy softmax-based objective would silently
misinterpret it).

Rather than re-litigate the ordinal encoding, rev-8 adds a NEW loss module,
`PyramidAuxiliaryLoss` (src/models/structured_losses.py), that supervises
ONLY boundary/direction/t_mse/segment -- zero count-loss term -- and is
composed ADDITIVELY on top of the unmodified legacy `ZipCountLoss` total via
new `loss.aux_lambda_*` keys that ZipCountLoss never reads (train.py, "legacy"
branch). Phase b on failure no longer stops the chain: it records the failure
and continues into `caux -> c2aux -> d1aux -> d2aux`, which mirror the retired
c/c2/d1/d2 mutations' lambda values (0.10/0.05/0.05/0.05) but write
`aux_lambda_*` instead of `lambda_*`, so the count objective and every
existing loss term (emae/consistency/monotonic/dice/vad/overlap/smooth) never
changes. b2/c/c2/d1/d2/e and structured_loss_block remain in this file,
exercised only by tests/test_phase8_driver.py's direct regression tests, not
by ORDER/GENERIC_MUTS/main() -- stage-2 (formerly phase e) is deliberately
NOT re-added yet; whether it is worth an "eaux" phase depends on what (if
anything) survives caux/c2aux/d1aux/d2aux.

Revision-7 (user-approved after the pyramid mask transfer test): the base
config carries `stack_input_mask: [1, 1, 1, 0, 0, 0]` for every arm a0->e.
Evidence chain: the matched stack-mask graduation (TCN family, 3 paired
seeds: pre012 vs all6 +0.0043, adopt via R3; its all6 arms replicated the
p8a_average root to six decimals) and the pyramid transfer test
(PYRAMID_MASK_TRANSFER_PREREG.md, chain-a0 configs +/- mask: +0.0073, passes
R2 AND R3, strict-NI, dOSD +0.032).  Without the mask the pyramid-minimal a0
sits BELOW the TCN root on every seed (-0.0067); with it, at parity.  The
mask is parameter/init-matched (no module or state removed) and the final
+0.010 complexity gate still references the immutable TCN root.

Revision-4/5 faults fixed here:

  BL1  The ladder compared each challenger against the CURRENT HOLDER, so a
       failing rung could be jumped (cap fails at +0.0049, stage2 then beats
       carried directly at +0.0050 while its true refinement over cap is
       +0.0001).  The ladder is now STRICT with fixed adjacent contrasts —
       cap-vs-carried, noDS-vs-cap, DS-vs-noDS — and the first failing rung
       ENDS the ladder.  No rung can be promoted without its own direct
       factor contrast passing the practical bar.
  BL2  seal_root() was fail-open: subset verification accepted an empty or
       truncated manifest, missing checkpoints sealed as null==null, and the
       first seal raced non-atomically.  Sealing is now a SEPARATE reviewed
       command (--seal-root): exact key sets over results+checkpoints+configs,
       every file must exist and be non-empty, no sha may be null, creation is
       atomic-exclusive (tmp + os.link), and the chain REFUSES to run without
       an existing, fully-verified manifest.  A process-wide flock prevents
       two drivers from interleaving.
  H    Fingerprints are fail-closed: a missing checkpoint or arm config
       raises instead of encoding None; seed identity is enforced (tags must
       end _s1234/_s2345/_s3456 in that exact order); root arm configs are
       resolved from the sealed P8A directory.  run_arm reuse now requires a
       lineage SIDECAR binding config sha + INIT sha + train/eval manifest
       shas + source-tree sha, so an identical YAML with a different INIT or
       evaluator can no longer be silently reused.  The completion sidecar
       also binds the result/checkpoint/config outputs; fresh runs re-read the
       YAML from disk after evaluation and require its input/output config
       hashes to agree before certification.
  M    final_promoted_config.yaml is written next to the candidate config (it
       is the ROOT config when the gate fails); final_state binds the sha of
       decisions.jsonl and any stale final from a superseded lineage is
       renamed on startup; the phase-e mechanism dump runs on the CHOSEN arm;
       the boundary protocol sha covers the evaluator AND its scoring
       dependencies.

The training-side `require_backbone_complete` guard is enforced by train.py;
chain arms therefore fail before training if backbone-only initialization is
incomplete.
"""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/home/edabk/miniconda3/envs/.zipformer/bin/python"
INIT = "logs/r8_v3_diverse/best_macro_f1.pt"
V4_CONFIG = "configs/zipcount_v4_distill.yaml"
PYR_CONFIG = "configs/zipcount_pyramid_ordinal.yaml"
VOX_SEL = "/media/edabk/500GB Hard Disk/data_diar/voxconverse/vox_sel.json"
TEACHER_DIR = "/media/edabk/500GB Hard Disk/data_diar/voxconverse/diarizen_on_test"
ART = "artifacts/phase8_chain"
ROOT_ART = "artifacts/phase8a_alignment"
SEEDS = (1234, 2345, 3456)
STEPS = 3000
NEED_GB = 16
NONINF_MARGIN = -0.002
BOUNDARY_AP_GATE = 0.05
BOUNDARY_RECALL_GAIN_GATE = 0.05
FINAL_ROOT_GATE = 0.010

ROOT_RESULTS = [f"results/p8a_average_s{seed}_voxsel.json" for seed in SEEDS]
ROOT_CHECKPOINTS = [f"logs/p8a_average_s{seed}/step3000.pt" for seed in SEEDS]
ROOT_CONFIGS = [f"{ROOT_ART}/p8a_average_s{seed}.yaml" for seed in SEEDS]

# Everything whose behaviour shapes a training/eval outcome.  A change to any
# of these invalidates arm reuse (H: sidecar binding).
ZIPFORMER_DIR = "third_party/icefall/egs/librispeech/ASR/zipformer"
SOURCE_FILES = [
    "src/train.py",
    "src/models/heads.py",
    "src/models/pyramid_head.py",
    "src/models/losses.py",
    "src/models/structured_losses.py",
    "src/models/zipcount_v1.py",
    "src/models/zipformer_wrapper.py",
    "src/data/dataset.py",
    "src/data/label_utils.py",
    "src/data/feature_extractor.py",
    "scripts/eval_per_recording.py",
    # dynamically-imported encoder recipe (6th review: the source-tree sha
    # must cover the execution path, not only first-party files)
    f"{ZIPFORMER_DIR}/zipformer.py",
    f"{ZIPFORMER_DIR}/scaling.py",
    f"{ZIPFORMER_DIR}/subsampling.py",
    f"{ZIPFORMER_DIR}/model.py",
    f"{ZIPFORMER_DIR}/finetune.py",
    # Imported by finetune.py on the model-construction path.  In particular,
    # decoder/joiner initialization consumes RNG before the unused RNNT
    # branches are dropped, so source drift here can change count-head init.
    f"{ZIPFORMER_DIR}/encoder_interface.py",
    f"{ZIPFORMER_DIR}/decoder.py",
    f"{ZIPFORMER_DIR}/joiner.py",
    f"{ZIPFORMER_DIR}/optim.py",
    f"{ZIPFORMER_DIR}/asr_datamodule.py",
]
BOUNDARY_PROTOCOL_FILES = [
    "scripts/eval_boundary_ap.py",
    "src/models/structured_losses.py",
    "src/data/label_utils.py",
    "src/data/dataset.py",
]

STACK_DIMS = [192, 256, 384, 512, 384, 256]
FINAL_DIM = 512


# --------------------------------------------------------------------------
# configs
# --------------------------------------------------------------------------

def base_config() -> dict:
    config = yaml.safe_load(open(os.path.join(REPO, PYR_CONFIG)))
    v4 = yaml.safe_load(open(os.path.join(REPO, V4_CONFIG)))
    enc = config["model"]["encoder"]
    enc["multiscale_alignment"] = "average"
    enc["multiscale_include_final"] = False
    config["model"]["head"] = {
        "type": "gated_pyramid_ordinal",
        "d_model": 192,
        "dilations": [1, 2, 4, 8, 16, 32],
        "kernel": 3,
        "dropout": 0.1,
        "gate_mode": "global",
        "ordinal_mode": "softmax",
        "edge_hidden_dim": 96,
        "use_stage2": False,
        # Rev-7: pre-registered pre-bottleneck view, proven on both head
        # families (see module docstring).  Applies to every arm so all
        # adjacent contrasts stay mask-matched.
        "stack_input_mask": [1, 1, 1, 0, 0, 0],
    }
    config["loss"] = copy.deepcopy(v4["loss"])
    config["loss"]["type"] = "legacy"
    config["loss"]["lambda_boundary_smooth"] = 0.0
    config["loss"]["lambda_event"] = 0.0
    config["loss"]["lambda_raw_anchor"] = 0.0
    config["distill"]["teacher_dir"] = ""
    config["distill"]["lambda_kd"] = 0.0
    config["data"]["train_manifest"] = (
        "/media/edabk/500GB Hard Disk/data_diar/manifests/train_manifest_v3.json"
    )
    config["data"]["val_manifest"] = (
        "/home/edabk/hoangbpm/diar/diar_new/data/meetings/val_manifest.json"
    )
    return config


def structured_loss_block(count_loss_type: str) -> dict:
    return {
        "type": "pyramid_structured",
        "count_loss_type": count_loss_type,
        "sord_alpha": 2.5,
        "class_weights": [1.0, 1.0, 1.5, 3.0],
        "lambda_stage1": 0.0,
        "lambda_boundary": 0.0,
        "lambda_direction": 0.0,
        "lambda_t_mse": 0.0,
        "lambda_segment": 0.0,
        "lambda_delta": 0.0,
        "boundary_kernel": [0.25, 0.75, 1.0, 0.75, 0.25],
        "t_mse_tau": 4.0,
        "direction_class_weights": [1.0, 1.0],
        "lambda_smooth": 0.0,
        "lambda_boundary_smooth": 0.0,
        "lambda_vad": 0.0,
        "lambda_overlap": 0.0,
        "lambda_emae": 0.0,
        "lambda_consistency": 0.0,
        "lambda_monotonic": 0.0,
        "lambda_dice": 0.0,
        "lambda_event": 0.0,
        "lambda_raw_anchor": 0.0,
    }


def mut_a1(config):
    """Make fusion gates frame-dependent; do not add another feature source."""
    config["model"]["head"]["gate_mode"] = "framewise"


def mut_a2(config):
    """Add the encoder's final output as a residual after gate mode is settled."""
    config["model"]["encoder"]["multiscale_include_final"] = True


def mut_b_sord(config):
    config["loss"] = structured_loss_block("sord")


def mut_b_cum(config):
    config["loss"] = structured_loss_block("cumulative")


def mut_b2(config):
    config["model"]["head"]["ordinal_mode"] = "corn"
    config["loss"]["count_loss_type"] = "corn"


def mut_c(config):
    config["loss"]["lambda_boundary"] = 0.10


def mut_c2(config):
    config["loss"]["lambda_direction"] = 0.05


def mut_d1(config):
    config["loss"]["lambda_t_mse"] = 0.05


def mut_d2(config):
    config["loss"]["lambda_segment"] = 0.05


# --------------------------------------------------------------------------
# rev-8 legacy-bridge mutations: boundary/direction/t_mse/segment as ADDITIVE
# terms on the unchanged legacy loss (PyramidAuxiliaryLoss), not as a switch
# to pyramid_structured/SORD/CORN.  Phase b's structured bridge failed its
# own non-inferiority bar both ways (sord -0.0087, cumulative -0.0122),
# which structurally blocked mut_b2/mut_c/mut_c2/mut_d1/mut_d2/mut_e_stage2_*
# above -- they are increments on structured_loss_block and are landmines
# under a still-"legacy" carried_cfg (mut_b2 in particular: CORN mode makes
# the model emit count_logits already in LOG-PROBABILITY space, which the
# legacy loss's softmax-based objective would silently misinterpret).
# These mutations instead touch only loss.aux_lambda_* (see train.py), which
# ZipCountLoss never reads and PyramidAuxiliaryLoss adds independently of
# the count objective -- safe to compound onto carried_cfg regardless of
# what phase b did.
# --------------------------------------------------------------------------

def mut_caux(config):
    config["loss"]["aux_lambda_boundary"] = 0.10


def mut_c2aux(config):
    config["loss"]["aux_lambda_direction"] = 0.05


def mut_d1aux(config):
    config["loss"]["aux_lambda_t_mse"] = 0.05


def mut_d2aux(config):
    config["loss"]["aux_lambda_segment"] = 0.05


def mut_e_stage2_nods(config):
    config["model"]["head"]["use_stage2"] = True
    config["loss"]["lambda_stage1"] = 0.0


def mut_e_stage2_ds(config):
    config["model"]["head"]["use_stage2"] = True
    config["loss"]["lambda_stage1"] = 0.20


def head_param_count(head_cfg: dict, include_final: bool) -> int:
    from src.models.pyramid_head import GatedPyramidOrdinalHead
    final = FINAL_DIM if include_final else None
    head = GatedPyramidOrdinalHead(
        d_in=sum(STACK_DIMS) + (final or 0),
        stack_dims=STACK_DIMS,
        final_dim=final,
        d_model=int(head_cfg["d_model"]),
        dilations=tuple(head_cfg.get("dilations", [1, 2, 4, 8, 16, 32])),
        kernel=int(head_cfg.get("kernel", 3)),
        gate_mode=head_cfg.get("gate_mode"),
        ordinal_mode=head_cfg.get("ordinal_mode", "softmax"),
        edge_hidden_dim=int(head_cfg.get("edge_hidden_dim", 96)),
        use_stage2=bool(head_cfg.get("use_stage2", False)),
    )
    return sum(p.numel() for p in head.parameters())


def matched_cap_width(carried_cfg: dict) -> Tuple[int, float]:
    include_final = bool(
        carried_cfg["model"]["encoder"].get("multiscale_include_final", False))
    stage2_cfg = copy.deepcopy(carried_cfg["model"]["head"])
    stage2_cfg["use_stage2"] = True
    target = head_param_count(stage2_cfg, include_final)
    best_width, best_err = None, None
    for width in range(int(carried_cfg["model"]["head"]["d_model"]), 512):
        one_cfg = copy.deepcopy(carried_cfg["model"]["head"])
        one_cfg["use_stage2"] = False
        one_cfg["d_model"] = width
        count = head_param_count(one_cfg, include_final)
        err = abs(count - target) / target
        if best_err is None or err < best_err:
            best_width, best_err = width, err
        if count > target * 1.05:
            break
    assert best_err is not None and best_err <= 0.01, (
        f"no one-stage width matches stage-2 params within 1% "
        f"(best d={best_width}, err={best_err:.3%})")
    return best_width, best_err


def mut_e_cap_factory(width: int):
    def mut(config):
        config["model"]["head"]["d_model"] = int(width)
    return mut


# --------------------------------------------------------------------------
# hashing / fingerprints (fail-closed)
# --------------------------------------------------------------------------

def canonical(config: dict) -> str:
    return yaml.safe_dump(config, sort_keys=True)


def sha_of_config(config: dict) -> str:
    return hashlib.sha256(canonical(config).encode()).hexdigest()


def sha_of_file(path: str, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def combined_source_sha(rel_paths: List[str]) -> str:
    digest = hashlib.sha256()
    for rel in rel_paths:
        digest.update(rel.encode())
        digest.update(sha_of_file(os.path.join(REPO, rel)).encode())
    return digest.hexdigest()


def tag_of_result(result_path: str) -> str:
    name = os.path.basename(result_path)
    assert name.endswith("_voxsel.json"), result_path
    return name[: -len("_voxsel.json")]


def arm_config_path(tag: str) -> str:
    """Root arms resolve ONLY from the sealed P8A directory and chain arms
    ONLY from ART — a same-named file in the other directory can never
    shadow (5th review)."""
    base = ROOT_ART if tag.startswith("p8a_average_") else ART
    candidate = os.path.join(REPO, base, f"{tag}.yaml")
    if not os.path.exists(candidate):
        raise SystemExit(f"FAIL-CLOSED: no arm config for {tag} in {base}")
    return candidate


def result_fingerprint(result_path: str) -> dict:
    """Fail-closed identity: every component must exist and hash."""
    tag = tag_of_result(result_path)
    full = os.path.join(REPO, result_path)
    if not os.path.exists(full) or os.path.getsize(full) == 0:
        raise SystemExit(f"FAIL-CLOSED: result missing/empty: {result_path}")
    ckpt = os.path.join(REPO, f"logs/{tag}/step{STEPS}.pt")
    if not os.path.exists(ckpt) or os.path.getsize(ckpt) == 0:
        raise SystemExit(f"FAIL-CLOSED: checkpoint missing/empty for {tag}")
    return {
        "result": result_path,
        "result_sha": sha_of_file(full),
        "checkpoint_sha": sha_of_file(ckpt),
        "arm_config_sha": sha_of_config(
            yaml.safe_load(open(arm_config_path(tag)))),
    }


def assert_seed_identity(results: List[str], where: str) -> None:
    """Exactly SEEDS, unique, in order — not merely three entries."""
    suffixes = [f"_s{seed}" for seed in SEEDS]
    if len(results) != len(SEEDS):
        raise SystemExit(
            f"LINEAGE MISMATCH ({where}): expected {len(SEEDS)} results, "
            f"got {len(results)}")
    prefixes = set()
    for path, suffix in zip(results, suffixes):
        tag = tag_of_result(path)
        if not tag.endswith(suffix):
            raise SystemExit(
                f"LINEAGE MISMATCH ({where}): {path} is not the {suffix} "
                f"arm — seed identity/order violated")
        prefixes.add(tag[: -len(suffix)])
    if len(prefixes) != 1:
        raise SystemExit(
            f"LINEAGE MISMATCH ({where}): seeds come from different arms "
            f"{sorted(prefixes)} — a result vector must be one arm family")


def state_fingerprint(config: dict, results: List[str]) -> dict:
    assert_seed_identity(results, "state_fingerprint")
    return {"config_sha": sha_of_config(config),
            "results": [result_fingerprint(p) for p in results]}


def verify_state(expected: dict, config: dict, results: List[str],
                 where: str) -> None:
    if expected.get("config_sha") != sha_of_config(config):
        raise SystemExit(f"LINEAGE MISMATCH ({where}): config sha changed")
    assert_seed_identity(results, where)
    entries = expected.get("results", [])
    if len(entries) != len(SEEDS):
        raise SystemExit(
            f"LINEAGE MISMATCH ({where}): expected {len(SEEDS)} result "
            f"entries in the record, found {len(entries)}")
    for entry, path in zip(entries, results):
        if entry.get("result") != path:
            raise SystemExit(
                f"LINEAGE MISMATCH ({where}): result path {path} != "
                f"recorded {entry.get('result')}")
        live = result_fingerprint(path)
        for key in ("result_sha", "checkpoint_sha", "arm_config_sha"):
            if entry.get(key) is None or entry.get(key) != live[key]:
                raise SystemExit(
                    f"LINEAGE MISMATCH ({where}): {key} drifted for {path}")


# --------------------------------------------------------------------------
# root sealing (BL2: fail-closed, atomic, reviewed)
# --------------------------------------------------------------------------

def _root_live_state() -> dict:
    live = {"results": {}, "checkpoints": {}, "configs": {}}
    for kind, paths in (("results", ROOT_RESULTS),
                        ("checkpoints", ROOT_CHECKPOINTS),
                        ("configs", ROOT_CONFIGS)):
        for rel in paths:
            full = os.path.join(REPO, rel)
            if not os.path.exists(full) or os.path.getsize(full) == 0:
                raise SystemExit(
                    f"ROOT SEAL FAIL-CLOSED: {rel} is missing or empty")
            live[kind][rel] = sha_of_file(full)
    return live


def seal_root_command(manifest_path: str) -> None:
    """Reviewed one-shot action: atomically create the root manifest."""
    if os.path.exists(manifest_path):
        raise SystemExit(f"refusing to reseal: {manifest_path} already exists")
    live = _root_live_state()
    tmp = manifest_path + f".tmp.{os.getpid()}"
    with open(tmp, "w") as sink:
        json.dump({**live, "sealed_at": time.strftime("%F %T")}, sink, indent=1)
        sink.flush()
        os.fsync(sink.fileno())
    try:
        os.link(tmp, manifest_path)     # atomic + exclusive
    except FileExistsError:
        raise SystemExit("another process sealed the root concurrently")
    finally:
        os.unlink(tmp)
    verify_root(manifest_path)
    print(f"[root] sealed {sum(len(v) for k, v in live.items() if k != 'sealed_at')} "
          f"artifacts into {os.path.basename(manifest_path)}")


def verify_root(manifest_path: str) -> None:
    if not os.path.exists(manifest_path):
        raise SystemExit(
            "root manifest missing — run the reviewed seal step first:\n"
            "  PYTHONPATH=. python scripts/phase8_chain_driver.py --seal-root")
    sealed = json.load(open(manifest_path))
    expected_keys = {"results": set(ROOT_RESULTS),
                     "checkpoints": set(ROOT_CHECKPOINTS),
                     "configs": set(ROOT_CONFIGS)}
    live = _root_live_state()
    for kind, expected in expected_keys.items():
        sealed_kind = sealed.get(kind, {})
        if set(sealed_kind) != expected:
            raise SystemExit(
                f"ROOT VIOLATION: manifest {kind} keys "
                f"{sorted(sealed_kind)} != expected {sorted(expected)}")
        for rel, sha in sealed_kind.items():
            if not sha:
                raise SystemExit(f"ROOT VIOLATION: null sha sealed for {rel}")
            if live[kind][rel] != sha:
                raise SystemExit(
                    f"ROOT VIOLATION: {rel} no longer matches the sealed "
                    "manifest")


def acquire_process_lock() -> "os.file":  # noqa: F821 - descriptive
    lock_path = os.path.join(REPO, ART, "chain.lock")
    handle = open(lock_path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(
            "another phase8_chain_driver is already running (chain.lock held)")
    handle.write(f"pid={os.getpid()} at={time.strftime('%F %T')}\n")
    handle.flush()
    return handle


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def check_disk() -> None:
    free_gb = shutil.disk_usage("/").free // (1 << 30)
    if free_gb < NEED_GB:
        raise SystemExit(f"FATAL: only {free_gb} GiB free on /, need {NEED_GB}")


def finalize_training_block(config: dict, seed: int, tag: str) -> None:
    config["training"].update(
        {
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
            "log_dir": f"logs/{tag}",
        }
    )
    config["training"].pop("backbone_lr", None)
    config["training"].pop("head_lr", None)


def arm_inputs(config: dict) -> dict:
    """Everything an arm's outcome depends on beyond its YAML (H)."""
    return {
        "config_sha": sha_of_config(config),
        "init_sha": sha_of_file(os.path.join(REPO, INIT)),
        "train_manifest_sha": sha_of_file(config["data"]["train_manifest"]),
        "eval_manifest_sha": sha_of_file(VOX_SEL),
        "source_sha": combined_source_sha(SOURCE_FILES),
    }


def arm_outputs(tag: str, result_rel: str) -> dict:
    return {
        "result_sha": sha_of_file(os.path.join(REPO, result_rel)),
        "checkpoint_sha": sha_of_file(
            os.path.join(REPO, f"logs/{tag}/step{STEPS}.pt")),
        "arm_config_sha": sha_of_config(
            yaml.safe_load(open(os.path.join(REPO, ART, f"{tag}.yaml")))),
    }


def _invalidate(tag: str, stamp: str) -> None:
    for path in (
        os.path.join(REPO, f"results/{tag}_voxsel.json"),
        os.path.join(REPO, ART, f"{tag}.yaml"),
        os.path.join(REPO, ART, f"{tag}.lineage.json"),
        os.path.join(REPO, ART, f"{tag}_boundary_ap.json"),
        os.path.join(REPO, ART, f"{tag}_segments.txt"),
    ):
        if os.path.exists(path):
            os.rename(path, f"{path}.stale.{stamp}")


def run_arm(phase: str, arm: str, config: dict, seed: int) -> str:
    tag = f"p8{phase}_{arm}_s{seed}"
    result_rel = f"results/{tag}_voxsel.json"
    result = os.path.join(REPO, result_rel)
    cfg_path = os.path.join(REPO, ART, f"{tag}.yaml")
    sidecar_path = os.path.join(REPO, ART, f"{tag}.lineage.json")

    config = copy.deepcopy(config)
    finalize_training_block(config, seed, tag)
    wanted_yaml = canonical(config)
    wanted_inputs = arm_inputs(config)

    if os.path.exists(result):
        stored_yaml = (open(cfg_path).read() if os.path.exists(cfg_path) else None)
        stored = (json.load(open(sidecar_path))
                  if os.path.exists(sidecar_path) else None)
        # 5th review: reuse requires the OUTPUTS to still match too — an
        # edited result.json or regenerated checkpoint under an unchanged
        # YAML/inputs must never be laundered through a [skip].
        live_ok = False
        if stored_yaml == wanted_yaml and stored is not None \
                and stored.get("inputs") == wanted_inputs:
            try:
                live_ok = stored.get("outputs") == arm_outputs(tag, result_rel)
            except SystemExit:
                live_ok = False
        if live_ok:
            print(f"    [skip] {tag} (inputs + outputs sidecar verified)",
                  flush=True)
            return result_rel
        stamp = time.strftime("%Y%m%d%H%M%S")
        _invalidate(tag, stamp)
        print(f"    [rerun] {tag}: config/lineage/output drift — artifacts -> "
              f"*.stale.{stamp}", flush=True)

    check_disk()
    with open(cfg_path, "w") as stream:
        stream.write(wanted_yaml)
    print(f"    [run ] {tag} sha={wanted_inputs['config_sha'][:12]}", flush=True)

    log_dir = os.path.join(REPO, f"logs/{tag}")
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    t0 = time.time()
    with open(os.path.join(REPO, f"logs/{tag}.log"), "w") as log:
        subprocess.run(
            [PY, "src/train.py", "--config", cfg_path, "--init-from", INIT],
            cwd=REPO, env={**os.environ, "PYTHONPATH": "."},
            stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    for name in os.listdir(log_dir):
        if name != f"step{STEPS}.pt" and not name.startswith("events"):
            os.remove(os.path.join(log_dir, name))
    subprocess.run(
        [PY, "scripts/eval_per_recording.py",
         "--manifest", VOX_SEL,
         "--checkpoint", f"logs/{tag}/step{STEPS}.pt",
         "--config", cfg_path,
         "--out", result_rel],
        cwd=REPO, env={**os.environ, "PYTHONPATH": "."}, check=True,
    )
    # TOCTOU guard: re-read the actual YAML and re-hash every input AFTER
    # training+eval.  Hashing only the in-memory config would miss an external
    # edit to cfg_path while the arm was running.
    try:
        yaml_after = open(cfg_path).read()
    except OSError as exc:
        raise SystemExit(
            f"{tag}: config YAML disappeared WHILE the arm was running — "
            "the run is not certified") from exc
    if yaml_after != wanted_yaml:
        raise SystemExit(
            f"{tag}: config YAML changed WHILE the arm was running — "
            "the run is not certified")
    config_after = yaml.safe_load(yaml_after)
    inputs_after = arm_inputs(config_after)
    if inputs_after != wanted_inputs:
        raise SystemExit(
            f"{tag}: inputs changed WHILE the arm was running "
            "(init/manifest/source drift) — the run is not attributable to "
            "either revision and is not certified")
    outputs_after = arm_outputs(tag, result_rel)
    if outputs_after["arm_config_sha"] != wanted_inputs["config_sha"]:
        raise SystemExit(
            f"{tag}: input/output config SHA mismatch after evaluation — "
            "the run is not certified")
    # the sidecar is written LAST and atomically: its existence certifies a
    # completed, attributable run with these exact inputs and outputs
    payload = {"inputs": wanted_inputs, "outputs": outputs_after}
    tmp_sidecar = sidecar_path + f".tmp.{os.getpid()}"
    with open(tmp_sidecar, "w") as sink:
        json.dump(payload, sink, indent=1)
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(tmp_sidecar, sidecar_path)
    print(f"    [done] {tag} in {(time.time()-t0)/60:.1f} min", flush=True)
    return result_rel


def train_arm_all_seeds(phase: str, arm: str, config: dict) -> List[str]:
    return [run_arm(phase, arm, config, seed) for seed in SEEDS]


# --------------------------------------------------------------------------
# statistics + decision logic
# --------------------------------------------------------------------------

def compare(a_paths: List[str], b_paths: List[str]) -> dict:
    A = [json.load(open(os.path.join(REPO, p))) for p in a_paths]
    B = [json.load(open(os.path.join(REPO, p))) for p in b_paths]
    sets = [frozenset(r["per_recording"]) for r in A + B]
    if len(set(sets)) != 1:
        sizes = sorted({len(s) for s in sets})
        raise ValueError(
            f"recording sets differ across results (sizes {sizes}) — a paired "
            "comparison over unequal corpora is invalid; regenerate the arms")
    recs = sorted(sets[0])
    per_seed = [b["pooled"]["macro_f1"] - a["pooled"]["macro_f1"]
                for a, b in zip(A, B)]
    d_f12 = float(np.mean([b["pooled"]["f1_2"] - a["pooled"]["f1_2"]
                           for a, b in zip(A, B)]))
    d_osd = float(np.mean([b["pooled"]["osd_f1"] - a["pooled"]["osd_f1"]
                           for a, b in zip(A, B)]))
    da = np.array([[r["per_recording"][x]["macro_f1"] for x in recs] for r in A]).mean(0)
    db = np.array([[r["per_recording"][x]["macro_f1"] for x in recs] for r in B]).mean(0)
    d = db - da
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(recs), size=(10000, len(recs)))].mean(1)
    return {
        "per_seed": [float(x) for x in per_seed],
        "mean_pooled_delta": float(np.mean(per_seed)),
        "delta_f1_2": d_f12,
        "delta_osd": d_osd,
        "boot_mean": float(d.mean()),
        "ci": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
        "n_improved": int((d > 0).sum()),
        "n_recordings": len(recs),
    }


def evidence_label(stats: dict) -> str:
    if all(x > 0 for x in stats["per_seed"]) and stats["ci"][0] > 0:
        return "REAL"
    if all(x < 0 for x in stats["per_seed"]) and stats["ci"][1] < 0:
        return "NEGATIVE"
    if stats["mean_pooled_delta"] > 0:
        return "SUGGESTIVE"
    return "NONE"


def adoption(stats: dict) -> Tuple[bool, str]:
    if stats["mean_pooled_delta"] >= 0.005:
        return True, "R2 mean pooled delta >= +0.005"
    if (stats["delta_f1_2"] >= 0.010 or stats["delta_osd"] >= 0.010) \
            and stats["mean_pooled_delta"] >= NONINF_MARGIN:
        return True, "R3 class-2/OSD >= +0.010 with macro within margin"
    return False, "below adoption bar"


def non_inferior(stats: dict) -> bool:
    return stats["mean_pooled_delta"] >= NONINF_MARGIN


def resolve_bridge(candidate: str, candidate_bridge: dict,
                   alternate: str, alternate_bridge: dict
                   ) -> Tuple[Optional[str], bool]:
    if non_inferior(candidate_bridge):
        return candidate, False
    if non_inferior(alternate_bridge):
        return alternate, True
    return None, False


def climb_ladder(start: str, rungs: List[str],
                 compare_fn: Callable[[str, str], dict]) -> Tuple[str, list]:
    """BL1: STRICT ladder with FIXED adjacent contrasts.

    Each challenger is compared against the immediately preceding rung — the
    scientific control for exactly its factor — and the FIRST failing rung
    ends the climb.  A later rung can therefore never be promoted on a
    carried-level comparison while its own factor contrast is millipoint.
    """
    holder = start
    previous = start
    trail = []
    for rung in rungs:
        stats = compare_fn(previous, rung)
        ok, rule = adoption(stats)
        trail.append({"challenger": rung, "control": previous,
                      "stats": stats, "adopted": ok, "rule": rule})
        if not ok:
            break                      # no jumping past a failed rung
        holder = rung
        previous = rung
    return holder, trail


def promote(gate_stats: Optional[dict]) -> Tuple[str, bool]:
    if gate_stats is None:
        return "root", True
    passed = gate_stats["mean_pooled_delta"] >= FINAL_ROOT_GATE
    return ("candidate" if passed else "root"), passed


# --------------------------------------------------------------------------
# boundary metrics
# --------------------------------------------------------------------------

def boundary_metrics(result_paths: List[str]) -> List[dict]:
    manifest_sha = sha_of_file(VOX_SEL)
    protocol_sha = combined_source_sha(BOUNDARY_PROTOCOL_FILES)
    out = []
    for path in result_paths:
        tag = tag_of_result(path)
        ckpt = os.path.join(REPO, f"logs/{tag}/step{STEPS}.pt")
        report = os.path.join(REPO, ART, f"{tag}_boundary_ap.json")
        keys = {"checkpoint_sha": sha_of_file(ckpt),
                "config_sha": sha_of_config(
                    yaml.safe_load(open(arm_config_path(tag)))),
                "manifest_sha": manifest_sha,
                "protocol_sha": protocol_sha}
        cached = None
        if os.path.exists(report):
            cached = json.load(open(report))
            if any(cached.get(k) != v for k, v in keys.items()):
                os.rename(report,
                          f"{report}.stale.{time.strftime('%Y%m%d%H%M%S')}")
                cached = None
        if cached is None:
            subprocess.run(
                [PY, "scripts/eval_boundary_ap.py",
                 "--manifest", VOX_SEL,
                 "--checkpoint", f"logs/{tag}/step{STEPS}.pt",
                 "--config", arm_config_path(tag),
                 "--checkpoint-sha", keys["checkpoint_sha"],
                 "--config-sha", keys["config_sha"],
                 "--protocol-sha", keys["protocol_sha"],
                 "--out", report],
                cwd=REPO, env={**os.environ, "PYTHONPATH": "."}, check=True,
            )
            cached = json.load(open(report))
            for k, v in keys.items():
                if cached.get(k) != v:
                    raise SystemExit(
                        f"boundary report post-validation failed for {tag}: "
                        f"{k} mismatch immediately after generation")
        out.append(cached)
    if len(out) != len(SEEDS):
        raise SystemExit("boundary metrics: wrong number of seed reports")
    return out


def assert_boundary_comparable(ctl: List[dict], trt: List[dict]) -> None:
    if len(ctl) != len(SEEDS) or len(trt) != len(SEEDS):
        raise SystemExit("boundary gate invalid: wrong report count")
    for c, t in zip(ctl, trt):
        for key in ("n_scored", "n_transitions", "fp_budget"):
            if c.get(key) != t.get(key):
                raise SystemExit(
                    f"boundary gate invalid: {key} differs between arms "
                    f"({c.get(key)} vs {t.get(key)})")


def mechanism(result_path: str) -> None:
    tag = tag_of_result(result_path)
    out = os.path.join(REPO, ART, f"{tag}_segments.txt")
    if os.path.exists(out):
        return
    tmp = out + ".tmp"
    try:
        with open(tmp, "w") as sink:
            subprocess.run(
                [PY, "scripts/segment_diagnosis.py",
                 "--manifest", VOX_SEL,
                 "--checkpoint", f"logs/{tag}/step{STEPS}.pt",
                 "--config", arm_config_path(tag),
                 "--teacher-dir", TEACHER_DIR],
                cwd=REPO, env={**os.environ, "PYTHONPATH": "."},
                stdout=sink, stderr=subprocess.STDOUT, check=True,
            )
        os.replace(tmp, out)
    except subprocess.CalledProcessError as error:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"    [warn] mechanism dump failed for {tag}: {error}", flush=True)


# --------------------------------------------------------------------------
# chain
# --------------------------------------------------------------------------

# rev-8: b2/c/c2/d1/d2/e (structured_loss_block increments) are RETIRED from
# the live ORDER -- phase b's structured bridge failed both ways, and running
# them under a still-"legacy" carried_cfg would be a silent no-op at best
# (unread config keys) and a landmine at worst (mut_b2's CORN mode). Their
# functions and structured_loss_block stay in this file, exercised only by
# tests/test_phase8_driver.py's direct regression tests, not by main().
ORDER = ["a0", "a1", "a2", "b", "caux", "c2aux", "d1aux", "d2aux"]
GENERIC_MUTS = {"a1": mut_a1, "a2": mut_a2,
                "caux": mut_caux, "c2aux": mut_c2aux,
                "d1aux": mut_d1aux, "d2aux": mut_d2aux}


def record(decisions_path: str, payload: dict, parent_state: dict,
           carried_cfg: dict, carried_results: List[str]) -> None:
    payload = {
        **payload,
        "parent_state": parent_state,
        "output_state": state_fingerprint(carried_cfg, carried_results),
        "carried_results": list(carried_results),
        "time": time.strftime("%F %T"),
    }
    with open(decisions_path, "a") as sink:
        sink.write(json.dumps(payload) + "\n")
    # any new decision supersedes previously written finals IMMEDIATELY — a
    # partial or bridge-failed run must not leave a stale canonical final
    stale_out_final_if_lineage_changed(decisions_path)


def replay(name: str, prior: dict, carried_cfg: dict,
           carried_results: List[str]) -> Tuple[dict, List[str]]:
    verify_state(prior["parent_state"], carried_cfg, carried_results,
                 where=f"replay {name}, parent")
    if not prior.get("adopt") and name in ("a0", "b"):
        raise SystemExit(f"recorded {name} decision was STOP; nothing to resume")
    if prior.get("adopt"):
        if name == "b":
            (mut_b_cum if prior["winner"] == "cumulative" else mut_b_sord)(carried_cfg)
        elif name == "e":
            chosen = prior["chosen"]
            if chosen == "stage2_ds":
                mut_e_stage2_ds(carried_cfg)
            elif chosen == "stage2_nods":
                mut_e_stage2_nods(carried_cfg)
            elif chosen == "cap":
                mut_e_cap_factory(prior["cap_width"])(carried_cfg)
        elif name != "a0":
            GENERIC_MUTS[name](carried_cfg)
        carried_results = list(prior["carried_results"])
    verify_state(prior["output_state"], carried_cfg, carried_results,
                 where=f"replay {name}, output")
    return carried_cfg, carried_results


def _lookup_decision(path: str, phase: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    best = None
    for line in open(path):
        parsed = json.loads(line)
        if parsed.get("phase") == phase:
            best = parsed
    return best


def report_stats(stats: dict, label: str, adopt: bool, rule: str) -> None:
    print(f"  per-seed deltas : {['%+.4f' % x for x in stats['per_seed']]}")
    print(f"  bootstrap       : {stats['boot_mean']:+.4f} "
          f"CI [{stats['ci'][0]:+.4f}, {stats['ci'][1]:+.4f}] "
          f"({stats['n_improved']}/{stats['n_recordings']} improved)")
    print(f"  class-2 / OSD   : {stats['delta_f1_2']:+.4f} / {stats['delta_osd']:+.4f}")
    print(f"  evidence        : {label}")
    print(f"  decision        : {'ADOPT' if adopt else 'not adopted'}  [{rule}]",
          flush=True)


def drift_vs_root(carried_results: List[str]) -> None:
    if carried_results == ROOT_RESULTS:
        return
    drift = compare(ROOT_RESULTS, carried_results)
    print(f"  drift vs TCN root: {drift['mean_pooled_delta']:+.4f} "
          f"(final gate: >= +{FINAL_ROOT_GATE})", flush=True)


def stale_out_final_if_lineage_changed(decisions_path: str) -> None:
    final_path = os.path.join(REPO, ART, "final_state.json")
    if not os.path.exists(final_path):
        return
    stored = json.load(open(final_path))
    live_sha = (sha_of_file(decisions_path)
                if os.path.exists(decisions_path) else None)
    if stored.get("decisions_sha") != live_sha:
        stamp = time.strftime("%Y%m%d%H%M%S")
        for name in ("final_state.json", "final_candidate_config.yaml",
                     "final_promoted_config.yaml"):
            path = os.path.join(REPO, ART, name)
            if os.path.exists(path):
                os.rename(path, f"{path}.stale.{stamp}")
        print(f"[final] previous final artifacts no longer match the decision "
              f"log — renamed *.stale.{stamp}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default=",".join(ORDER))
    ap.add_argument("--seal-root", action="store_true",
                    help="reviewed one-shot action: create the root manifest")
    args = ap.parse_args()

    os.makedirs(os.path.join(REPO, ART), exist_ok=True)
    manifest_path = os.path.join(REPO, ART, "root_manifest.json")
    if args.seal_root:
        seal_root_command(manifest_path)
        return

    wanted = [p for p in ORDER if p in {x.strip() for x in args.phases.split(",")}]
    if not wanted:
        raise SystemExit(f"no valid phases in {args.phases!r}; valid: {ORDER}")
    last_wanted = ORDER.index(wanted[-1])

    lock = acquire_process_lock()   # held for the driver's lifetime  # noqa: F841
    decisions_path = os.path.join(REPO, ART, "decisions.jsonl")
    verify_root(manifest_path)
    stale_out_final_if_lineage_changed(decisions_path)

    carried_cfg = base_config()
    carried_results = list(ROOT_RESULTS)

    for index, name in enumerate(ORDER):
        if index > last_wanted:
            break
        if name not in wanted:
            prior = _lookup_decision(decisions_path, name)
            if prior is None:
                raise SystemExit(f"phase {name} skipped but has no recorded decision")
            carried_cfg, carried_results = replay(
                name, prior, carried_cfg, carried_results)
            continue

        print(f"\n===== PHASE p8{name} =====", flush=True)
        parent_state = state_fingerprint(carried_cfg, carried_results)

        if name == "a0":
            trt = train_arm_all_seeds("a0", "pyr", carried_cfg)
            stats = compare(carried_results, trt)
            label = evidence_label(stats)
            ok = non_inferior(stats)
            report_stats(stats, label, ok,
                         f"bridge non-inferiority (mean >= {NONINF_MARGIN})")
            if ok:
                carried_results = trt
            record(decisions_path,
                   {"phase": name, "adopt": ok, "label": label, **stats},
                   parent_state, carried_cfg, carried_results)
            mechanism(trt[0])
            if not ok:
                print("\nBRIDGE FAILED — chain STOPPED.", flush=True)
                return
            drift_vs_root(carried_results)
            continue

        if name == "b":
            cfg_sord = copy.deepcopy(carried_cfg)
            mut_b_sord(cfg_sord)
            cfg_cum = copy.deepcopy(carried_cfg)
            mut_b_cum(cfg_cum)
            sord = train_arm_all_seeds("b", "sord", cfg_sord)
            cum = train_arm_all_seeds("b", "cum", cfg_cum)
            h2h = compare(sord, cum)
            cum_earned, h2h_rule = adoption(h2h)
            candidate, alternate = (("cumulative", "sord") if cum_earned
                                    else ("sord", "cumulative"))
            print(f"  head-to-head cum-vs-sord: {h2h['mean_pooled_delta']:+.4f} "
                  f"-> candidate {candidate}"
                  + (f" (earned: {h2h_rule})" if cum_earned else " (default)"))
            bridges = {"sord": compare(carried_results, sord),
                       "cumulative": compare(carried_results, cum)}
            winner, was_fallback = resolve_bridge(
                candidate, bridges[candidate], alternate, bridges[alternate])
            if winner is None:
                for arm_name in ("sord", "cumulative"):
                    print(f"  bridge {arm_name}: "
                          f"{bridges[arm_name]['mean_pooled_delta']:+.4f} "
                          f"(margin {NONINF_MARGIN})")
                record(decisions_path,
                       {"phase": name, "adopt": False, "label": "NONE",
                        "winner": None, "head_to_head": h2h,
                        "bridges": bridges},
                       parent_state, carried_cfg, carried_results)
                print(
                    "\nSTRUCTURED BRIDGE FAILED (both arms) — carried config "
                    "stays legacy/softmax. Continuing to the rev-8 "
                    "legacy-bridge phases (caux/c2aux/d1aux/d2aux), which "
                    "add boundary/direction/t_mse/segment on top of the "
                    "UNCHANGED legacy loss instead of requiring this switch.",
                    flush=True)
                continue
            stats = bridges[winner]
            label = evidence_label(stats)
            report_stats(stats, label, True,
                         f"structured bridge: {winner}"
                         + (" via symmetric fallback" if was_fallback else ""))
            carried_cfg = cfg_cum if winner == "cumulative" else cfg_sord
            carried_results = cum if winner == "cumulative" else sord
            record(decisions_path,
                   {"phase": name, "adopt": True, "label": label,
                    "winner": winner, "fallback": was_fallback,
                    "head_to_head": h2h, "bridges": bridges, **stats},
                   parent_state, carried_cfg, carried_results)
            mechanism(carried_results[0])
            drift_vs_root(carried_results)
            continue

        if name == "e":
            width, err = matched_cap_width(carried_cfg)
            print(f"  capacity arm width: d{width} (param err {err:.2%})")
            muts = {"cap": mut_e_cap_factory(width),
                    "stage2_nods": mut_e_stage2_nods,
                    "stage2_ds": mut_e_stage2_ds}
            cfgs, results = {}, {"carried": carried_results}

            def ensure_trained(arm_name: str) -> str:
                # LAZY (5th review): a rung is trained only when the ladder
                # actually reaches it — if cap fails its contrast, no stage-2
                # arm is ever trained, honouring "a failed rung stops the run",
                # not merely the comparison.
                if arm_name not in results:
                    cfg = copy.deepcopy(carried_cfg)
                    muts[arm_name](cfg)
                    cfgs[arm_name] = cfg
                    results[arm_name] = train_arm_all_seeds("e", arm_name, cfg)
                return arm_name

            chosen, trail = climb_ladder(
                "carried", ["cap", "stage2_nods", "stage2_ds"],
                lambda control, rung: compare(
                    results[control], results[ensure_trained(rung)]))
            for info in trail:
                print(f"  ladder {info['challenger']:>12} vs "
                      f"{info['control']:<12}: "
                      f"{info['stats']['mean_pooled_delta']:+.4f} -> "
                      f"{'CLIMB' if info['adopted'] else 'STOP'} "
                      f"[{info['rule']}]")
            print(f"  decision        : carry {chosen}", flush=True)
            if chosen != "carried":
                carried_cfg = cfgs[chosen]
                carried_results = results[chosen]
            record(decisions_path,
                   {"phase": name, "adopt": chosen != "carried",
                    "chosen": chosen, "cap_width": width, "ladder": trail},
                   parent_state, carried_cfg, carried_results)
            if chosen != "carried":
                mechanism(results[chosen][0])
            drift_vs_root(carried_results)
            continue

        cfg_trt = copy.deepcopy(carried_cfg)
        GENERIC_MUTS[name](cfg_trt)
        trt = train_arm_all_seeds(name, "trt", cfg_trt)
        stats = compare(carried_results, trt)
        label = evidence_label(stats)
        adopt, rule = adoption(stats)

        gate_payload = {}
        if name == "caux":
            ctl_metrics = boundary_metrics(carried_results)
            trt_metrics = boundary_metrics(trt)
            assert_boundary_comparable(ctl_metrics, trt_metrics)
            ap_mean = float(np.mean([m["boundary_ap"] for m in trt_metrics]))
            recall_gain = float(np.mean(
                [t["recall_at_fp_budget"] - c["recall_at_fp_budget"]
                 for t, c in zip(trt_metrics, ctl_metrics)]))
            gate_ok = (ap_mean >= BOUNDARY_AP_GATE
                       and recall_gain >= BOUNDARY_RECALL_GAIN_GATE)
            print(f"  boundary AP     : trt mean {ap_mean:.4f} "
                  f"(gate >= {BOUNDARY_AP_GATE})")
            print(f"  recall@FP-budget: gain {recall_gain:+.4f} "
                  f"(gate >= +{BOUNDARY_RECALL_GAIN_GATE})")
            gate_payload = {"boundary_ap_mean": ap_mean,
                            "recall_gain": recall_gain,
                            "boundary_gate_passed": gate_ok,
                            "ctl_boundary": ctl_metrics,
                            "trt_boundary": trt_metrics}
            if adopt and not gate_ok:
                adopt = False
                rule = "preregistered boundary gate failed — adoption vetoed"

        report_stats(stats, label, adopt, rule)
        if adopt:
            carried_cfg, carried_results = cfg_trt, trt
        record(decisions_path,
               {"phase": name, "adopt": adopt, "label": label, "rule": rule,
                **gate_payload, **stats},
               parent_state, carried_cfg, carried_results)
        mechanism(trt[0])
        drift_vs_root(carried_results)

    if last_wanted != len(ORDER) - 1:
        print(f"\nphase8 chain finished through p8{ORDER[last_wanted]} "
              "(partial run: no final promotion is written).", flush=True)
        return
    gate_stats = (None if carried_results == ROOT_RESULTS
                  else compare(ROOT_RESULTS, carried_results))
    promoted, passed = promote(gate_stats)
    promoted_cfg = (carried_cfg if promoted == "candidate"
                    else yaml.safe_load(open(os.path.join(REPO, ROOT_CONFIGS[0]))))
    final = {
        "candidate": state_fingerprint(carried_cfg, carried_results),
        "gate_stats": gate_stats,
        "final_gate_passed": passed,
        "promoted": promoted,
        "promoted_results": (carried_results if promoted == "candidate"
                             else list(ROOT_RESULTS)),
        "decisions_sha": sha_of_file(decisions_path),
    }
    if gate_stats is not None:
        verdict = ("PASSED: candidate promoted" if passed else
                   "NOT PASSED: the TCN root remains the promoted system")
        print(f"\nFINAL COMPLEXITY GATE vs TCN root: "
              f"{gate_stats['mean_pooled_delta']:+.4f} "
              f"(need >= +{FINAL_ROOT_GATE}) -> {verdict}")
    with open(os.path.join(REPO, ART, "final_candidate_config.yaml"), "w") as s:
        yaml.safe_dump(carried_cfg, s, sort_keys=False)
    with open(os.path.join(REPO, ART, "final_promoted_config.yaml"), "w") as s:
        yaml.safe_dump(promoted_cfg, s, sort_keys=False)
    with open(os.path.join(REPO, ART, "final_state.json"), "w") as s:
        json.dump(final, s, indent=1)
    print("\nphase8 chain COMPLETE; candidate + PROMOTED configs written "
          "(deploy from final_promoted_config.yaml only).", flush=True)


if __name__ == "__main__":
    main()
