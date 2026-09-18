#!/usr/bin/env bash
# Phase 8A: isolate the backbone-to-head temporal alignment defect.
#
# Only one factor changes:
#   average  -- exact historical avg_pool1d(2) multiscale alignment
#   learned  -- the pretrained Zipformer downsample_output applied to every
#               captured stack
#
# Both arms use the clean Phase-5c architecture/protocol: random TCN-192 head,
# identical v3 backbone, backbone frozen/eval, legacy losses unchanged, paired
# seeds/order/RNG, and locked step 3000.  The final-output residual, framewise
# gate, CORN, boundary losses, segment loss, and stage 2 are all deliberately
# absent here.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python
BASE_CFG=configs/zipcount_v4_distill.yaml
INIT=logs/r8_v3_diverse/best_macro_f1.pt
TRAIN_MANIFEST="/media/edabk/500GB Hard Disk/data_diar/manifests/train_manifest_v3.json"
VAL_MANIFEST="/home/edabk/hoangbpm/diar/diar_new/data/meetings/val_manifest.json"
VOX_SEL="/media/edabk/500GB Hard Disk/data_diar/voxconverse/vox_sel.json"
TEACHER_DIR="/media/edabk/500GB Hard Disk/data_diar/voxconverse/diarizen_on_test"
PHASE_DIR=artifacts/phase8a_alignment
HASH_FILE="$PHASE_DIR/preregister.sha256"
STEPS=3000
SEEDS=(1234 2345 3456)
ARMS=(average learned)

die() {
  echo "FATAL: $*" >&2
  exit 1
}

preflight() {
  [[ -x "$PY" ]] || die "missing Python environment: $PY"
  [[ -f "$BASE_CFG" ]] || die "missing base config: $BASE_CFG"
  [[ -s "$INIT" ]] || die "missing v3 initialization: $INIT"
  [[ -f "$TRAIN_MANIFEST" ]] || die "missing train manifest: $TRAIN_MANIFEST"
  [[ -f "$VAL_MANIFEST" ]] || die "missing validation manifest: $VAL_MANIFEST"
  [[ -f "$VOX_SEL" ]] || die "missing locked development manifest: $VOX_SEL"
  [[ -d "$TEACHER_DIR" ]] || die "missing mechanism teacher directory: $TEACHER_DIR"
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  command -v sha256sum >/dev/null || die "sha256sum is unavailable"

  "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("FATAL: CUDA is unavailable")
free, total = torch.cuda.mem_get_info()
if free < 20 * 1024**3:
    raise SystemExit(
        f"FATAL: need 20 GiB free GPU memory, found {free / 1024**3:.1f}"
    )
print(
    f"GPU preflight: {torch.cuda.get_device_name(0)}, "
    f"{free / 1024**3:.1f}/{total / 1024**3:.1f} GiB free"
)
PY

  local available_gb
  available_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  [[ "$available_gb" -ge 12 ]] \
    || die "need at least 12 GiB free on /, found ${available_gb} GiB"
  echo "Disk preflight: ${available_gb} GiB free"

  if pgrep -af 'src/train.py' >/dev/null; then
    die "another training process is already running"
  fi
  echo "Running repository regression tests..."
  PYTHONPATH=. "$PY" -m pytest -q tests \
    || die "repository regression tests failed"
  [[ ! -e "$PHASE_DIR" ]] || die "artifact target already exists: $PHASE_DIR"
  for seed in "${SEEDS[@]}"; do
    for arm in "${ARMS[@]}"; do
      local tag="p8a_${arm}_s${seed}"
      [[ ! -e "logs/$tag" ]] || die "run target already exists: logs/$tag"
      [[ ! -e "logs/${tag}.log" ]] || die "run log already exists: logs/${tag}.log"
      [[ ! -e "results/${tag}_voxsel.json" ]] \
        || die "result target already exists: results/${tag}_voxsel.json"
    done
  done
}

generate_config() {
  local arm=$1 seed=$2 tag=$3 output=$4
  "$PY" - "$output" "$arm" "$seed" "$tag" "$STEPS" <<'PY'
import sys
import yaml

output, arm, seed, tag, steps = sys.argv[1:6]
with open("configs/zipcount_v4_distill.yaml") as stream:
    config = yaml.safe_load(stream)

if arm not in {"average", "learned"}:
    raise ValueError(f"unknown Phase-8A arm: {arm}")

config["model"]["encoder"]["multiscale_alignment"] = (
    "average" if arm == "average" else "encoder_learned"
)
config["model"]["encoder"]["multiscale_include_final"] = False
config["model"]["head"] = {
    "type": "tcn_ordinal",
    "d_model": 192,
    "dropout": 0.1,
}

loss = config["loss"]
loss["type"] = "legacy"
loss["lambda_boundary_smooth"] = 0.0
loss["lambda_smooth"] = 0.02
loss["lambda_event"] = 0.0
loss["lambda_raw_anchor"] = 0.0

config["distill"]["teacher_dir"] = ""
config["distill"]["lambda_kd"] = 0.0

training = config["training"]
training.update(
    {
        "stage": "stage1_freeze_backbone",
        "freeze_backbone_all": True,
        "init_backbone_only": True,
        "reset_rng_after_init": True,
        "batch_size": 24,
        "lr": 1.0e-3,
        "lr_schedule": "cosine",
        "warmup_steps": 500,
        "max_steps": int(steps),
        "grad_clip": 5.0,
        "use_amp": True,
        "amp_init_scale": 4096.0,
        "amp_growth_interval": 2000,
        "num_workers": 12,
        "seed": int(seed),
        "eval_interval": 300,
        "eval_interval_early": 300,
        "early_phase_steps": int(steps),
        "save_every_eval": True,
        "snapshot_weights_only": True,
        "save_best": False,
        "save_last": False,
        "log_dir": f"logs/{tag}",
    }
)
training.pop("backbone_lr", None)
training.pop("head_lr", None)

config["data"]["train_manifest"] = (
    "/media/edabk/500GB Hard Disk/data_diar/manifests/train_manifest_v3.json"
)
config["data"]["val_manifest"] = (
    "/home/edabk/hoangbpm/diar/diar_new/data/meetings/val_manifest.json"
)

with open(output, "w") as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
PY
}

run_one() {
  local arm=$1 seed=$2
  local tag="p8a_${arm}_s${seed}"
  local cfg="$PHASE_DIR/${tag}.yaml"
  local run_dir="logs/$tag"
  local run_log="logs/${tag}.log"

  generate_config "$arm" "$seed" "$tag" "$cfg"
  sha256sum "$cfg" >> "$HASH_FILE"
  echo "[$(date --iso-8601=seconds)] START $tag"
  PYTHONPATH=. "$PY" src/train.py --config "$cfg" --init-from "$INIT" \
    > "$run_log" 2>&1

  [[ -s "$run_dir/step${STEPS}.pt" ]] \
    || die "$tag finished without locked step${STEPS}.pt"
  grep -aE "Multi-scale hypercolumn|Init weights|Reset model/augmentation RNG|Number of trainable|^\\[val@" \
    "$run_log" || true

  # These are immutable weight-only snapshots created by this run.  Keep only
  # the preregistered endpoint after successful completion to bound disk use.
  find "$run_dir" -maxdepth 1 -type f -name 'step*.pt' \
    ! -name "step${STEPS}.pt" -delete

  PYTHONPATH=. "$PY" scripts/eval_per_recording.py \
    --manifest "$VOX_SEL" \
    --checkpoint "$run_dir/step${STEPS}.pt" \
    --config "$cfg" \
    --out "results/${tag}_voxsel.json"
  echo "[$(date --iso-8601=seconds)] DONE $tag"
}

preflight
if [[ "${1:-}" == "--preflight-only" ]]; then
  echo "Phase-8A preflight passed; no artifacts were created."
  exit 0
fi
[[ $# -eq 0 ]] || die "usage: $0 [--preflight-only]"

mkdir -p "$PHASE_DIR" logs results
trap 'status=$?; if [[ $status -ne 0 ]]; then printf "%s\n" "FAILED status=$status at $(date --iso-8601=seconds)" > "$PHASE_DIR/FAILED"; fi' EXIT

{
  echo "Phase 8A learned temporal-alignment preregistration"
  echo "launched=$(date --iso-8601=seconds)"
  echo "primary=vox_sel recording macro-F1 at locked step $STEPS"
  echo "arms=exact legacy average vs pretrained encoder-learned alignment"
  echo "seeds=${SEEDS[*]}"
  echo "all later pyramid/CORN/boundary/stage2 changes are OFF"
} > "$PHASE_DIR/launch.txt"

sha256sum "$0" "$BASE_CFG" "$INIT" PYRAMID_ORDINAL_ROADMAP.md > "$HASH_FILE"

# Alternate arm order at the middle seed to balance machine-time order.
run_one average 1234
run_one learned 1234
run_one learned 2345
run_one average 2345
run_one average 3456
run_one learned 3456

"$PY" scripts/compare_runs.py \
  --a \
    results/p8a_average_s1234_voxsel.json \
    results/p8a_average_s2345_voxsel.json \
    results/p8a_average_s3456_voxsel.json \
  --b \
    results/p8a_learned_s1234_voxsel.json \
    results/p8a_learned_s2345_voxsel.json \
    results/p8a_learned_s3456_voxsel.json \
  --name-a "fixed-average" \
  --name-b "encoder-learned" \
  --metric macro_f1 | tee "$PHASE_DIR/comparison.txt"

for arm in average learned; do
  PYTHONPATH=. "$PY" scripts/segment_diagnosis.py \
    --manifest "$VOX_SEL" \
    --checkpoint "logs/p8a_${arm}_s1234/step${STEPS}.pt" \
    --config "$PHASE_DIR/p8a_${arm}_s1234.yaml" \
    --teacher-dir "$TEACHER_DIR" \
    > "$PHASE_DIR/${arm}_s1234_segments.txt"
done

printf "%s\n" "completed=$(date --iso-8601=seconds)" > "$PHASE_DIR/COMPLETE"
trap - EXIT
echo "Phase 8A complete."
