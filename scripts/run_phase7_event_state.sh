#!/usr/bin/env bash
# Phase 7: event-conditioned ordinal state filtering.
#
# Locked before seeing Phase-7 results (see EVENT_STATE_ROADMAP.md):
#   init       seed-matched p5c_tcn192 step3000 checkpoint
#   arms       C0 raw TCN; C1 raw TCN + signed event CE;
#              C2 event-conditioned causal filter + raw-emission anchor
#   regime     full-model FT, 800 steps, uniform 3e-5 LR, warmup 500,
#              legacy KL=0.02, KD=0, boundary T-MSE=0
#   primary    vox_sel recording-level macro-F1 at LOCKED step 400
#   controls   all three seeds use paired data order and reset post-init RNG;
#              C1/C2 have identical modules and parameter count
#
# The script is intentionally restart-conservative: it refuses any existing
# Phase-7 run/config/result target instead of deleting or mixing artifacts.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python
BASE_CFG=configs/zipcount_v4_distill.yaml
TRAIN_MANIFEST="/media/edabk/500GB Hard Disk/data_diar/manifests/train_manifest_v3.json"
VAL_MANIFEST="/home/edabk/hoangbpm/diar/diar_new/data/meetings/val_manifest.json"
VOX_SEL="/media/edabk/500GB Hard Disk/data_diar/voxconverse/vox_sel.json"
TEACHER_DIR="/media/edabk/500GB Hard Disk/data_diar/voxconverse/diarizen_on_test"
PHASE_DIR=artifacts/phase7_event_state
HASH_FILE="$PHASE_DIR/preregister.sha256"
COMPARE_FILE="$PHASE_DIR/step400_comparisons.txt"
STEPS=800
PRIMARY_STEP=400
SEEDS=(1234 2345 3456)
ARMS=(c0 c1 c2)

die() {
  echo "FATAL: $*" >&2
  exit 1
}

preflight() {
  [[ -x "$PY" ]] || die "missing Python environment: $PY"
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  command -v sha256sum >/dev/null || die "sha256sum is unavailable"
  [[ -f "$BASE_CFG" ]] || die "missing base config: $BASE_CFG"
  [[ -f "$TRAIN_MANIFEST" ]] || die "missing train manifest: $TRAIN_MANIFEST"
  [[ -f "$VAL_MANIFEST" ]] || die "missing validation manifest: $VAL_MANIFEST"
  [[ -f "$VOX_SEL" ]] || die "missing locked vox_sel manifest: $VOX_SEL"
  [[ -d "$TEACHER_DIR" ]] || die "missing diagnosis teacher directory: $TEACHER_DIR"

  for seed in "${SEEDS[@]}"; do
    [[ -s "logs/p5c_tcn192_s${seed}/step3000.pt" ]] \
      || die "missing init checkpoint for seed $seed"
  done

  "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("FATAL: CUDA is unavailable in the Zipformer environment")
free, total = torch.cuda.mem_get_info()
if free < 20 * 1024**3:
    raise SystemExit(f"FATAL: only {free / 1024**3:.1f} GiB GPU memory free")
print(f"GPU preflight: {torch.cuda.get_device_name(0)}, "
      f"{free / 1024**3:.1f}/{total / 1024**3:.1f} GiB free")
PY

  local available_gb
  available_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  [[ "$available_gb" -ge 20 ]] \
    || die "need at least 20 GiB free on /, found ${available_gb} GiB"
  echo "Disk preflight: ${available_gb} GiB free"

  [[ ! -e "$PHASE_DIR" ]] || die "artifact target already exists: $PHASE_DIR"
  for seed in "${SEEDS[@]}"; do
    for arm in "${ARMS[@]}"; do
      local tag="p7_${arm}_s${seed}"
      [[ ! -e "logs/$tag" ]] || die "run target already exists: logs/$tag"
      [[ ! -e "logs/${tag}.log" ]] || die "run log already exists: logs/${tag}.log"
      for step in 200 400 800; do
        [[ ! -e "results/${tag}_step${step}_voxsel.json" ]] \
          || die "result target already exists: results/${tag}_step${step}_voxsel.json"
      done
    done
  done
}

generate_config() {
  local arm=$1 seed=$2 tag=$3 output=$4
  "$PY" - "$output" "$arm" "$seed" "$tag" "$STEPS" <<'PY'
import sys
import yaml

output, arm, seed, tag, steps = sys.argv[1:6]
with open("configs/zipcount_v4_distill.yaml") as f:
    c = yaml.safe_load(f)

c["distill"]["lambda_kd"] = 0.0
c["distill"]["teacher_dir"] = ""
c["loss"]["lambda_boundary_smooth"] = 0.0
c["loss"]["lambda_smooth"] = 0.02
c["loss"]["event_class_weights"] = [12.0, 1.0, 12.0]

if arm == "c0":
    c["model"]["head"] = {
        "type": "tcn_ordinal", "d_model": 192, "dropout": 0.1,
    }
    c["loss"]["lambda_event"] = 0.0
    c["loss"]["lambda_raw_anchor"] = 0.0
    expected_missing = []
elif arm in ("c1", "c2"):
    c["model"]["head"] = {
        "type": "tcn_state_ordinal",
        "d_model": 192,
        "dropout": 0.1,
        "event_stay_bias": 2.9,
        "state_emission_scale": 1.0,
        "event_filter_class_weights": [12.0, 1.0, 12.0],
        "use_state_filter": arm == "c2",
    }
    c["loss"]["lambda_event"] = 0.2
    c["loss"]["lambda_raw_anchor"] = 0.25 if arm == "c2" else 0.0
    expected_missing = ["head.event_proj.weight", "head.event_proj.bias"]
else:
    raise ValueError(f"unknown arm {arm}")

t = c["training"]
t.update({
    "stage": "stage3_finetune",
    "freeze_backbone_all": False,
    "batch_size": 24,
    "lr": 3.0e-5,
    "backbone_lr": 3.0e-5,
    "head_lr": 3.0e-5,
    "lr_schedule": "cosine",
    "warmup_steps": 500,
    "max_steps": int(steps),
    "grad_clip": 5.0,
    "use_amp": True,
    # The recurrent filtered loss overflows fp16 activation gradients at the
    # PyTorch default 65536 scale on a real 400-frame batch.  4096 passed the
    # same end-to-end smoke with every one of 69 gradient tensors finite.
    "amp_init_scale": 4096.0,
    "amp_growth_interval": 2000,
    "num_workers": 12,
    "seed": int(seed),
    "eval_interval": 200,
    "eval_interval_early": 200,
    "early_phase_steps": int(steps),
    "save_every_eval": True,
    "snapshot_weights_only": True,
    "save_best": False,
    "save_last": True,
    "reset_rng_after_init": True,
    "init_backbone_only": False,
    "expected_init_missing": expected_missing,
    "require_no_unexpected_init_keys": True,
    "log_dir": f"logs/{tag}",
})

# Pin the manifests rather than inheriting a later local config edit.
c["data"]["train_manifest"] = \
    "/media/edabk/500GB Hard Disk/data_diar/manifests/train_manifest_v3.json"
c["data"]["val_manifest"] = \
    "/home/edabk/hoangbpm/diar/diar_new/data/meetings/val_manifest.json"

with open(output, "w") as f:
    yaml.safe_dump(c, f, sort_keys=False)
PY
}

run_one() {
  local arm=$1 seed=$2
  local tag="p7_${arm}_s${seed}"
  local cfg="$PHASE_DIR/${tag}.yaml"
  local init="logs/p5c_tcn192_s${seed}/step3000.pt"
  local run_dir="logs/$tag"
  local run_log="logs/${tag}.log"

  generate_config "$arm" "$seed" "$tag" "$cfg"
  sha256sum "$cfg" "$init" >> "$HASH_FILE"
  echo "[$(date --iso-8601=seconds)] START $tag"
  PYTHONPATH=. "$PY" src/train.py --config "$cfg" --init-from "$init" \
    > "$run_log" 2>&1

  for step in 200 400 600 800; do
    [[ -s "$run_dir/step${step}.pt" ]] \
      || die "$tag finished without step${step}.pt"
  done
  grep -aE "Init weights|Reset model/augmentation RNG|Event-state training|^\[val@(200|400|600|800)\]" \
    "$run_log" || true

  # `last.pt` contains the same step-800 weights plus a large optimizer state.
  # Once all four immutable weight snapshots exist, it is redundant and costs
  # ~0.75 GiB per arm. Delete only this exact generated file.
  [[ -s "$run_dir/last.pt" ]] || die "$tag has no resumable last.pt"
  rm -f -- "$run_dir/last.pt"
  echo "Removed redundant $run_dir/last.pt after verified completion."

  for step in 200 400 800; do
    PYTHONPATH=. "$PY" scripts/eval_per_recording.py \
      --manifest "$VOX_SEL" \
      --checkpoint "$run_dir/step${step}.pt" \
      --config "$cfg" \
      --out "results/${tag}_step${step}_voxsel.json"
  done
  echo "[$(date --iso-8601=seconds)] DONE $tag"
}

preflight
if [[ "${1:-}" == "--preflight-only" ]]; then
  echo "Phase-7 preflight passed; no files were created."
  exit 0
fi
[[ $# -eq 0 ]] || die "usage: $0 [--preflight-only]"

mkdir -p "$PHASE_DIR" logs results
trap 'status=$?; if [[ $status -ne 0 ]]; then printf "%s\n" "FAILED status=$status at $(date --iso-8601=seconds)" > "$PHASE_DIR/FAILED"; fi' EXIT

{
  echo "Phase 7 event-state preregistration"
  echo "launched=$(date --iso-8601=seconds)"
  echo "primary=vox_sel recording macro-F1 at step $PRIMARY_STEP"
  echo "seeds=${SEEDS[*]}"
  echo "arms=${ARMS[*]}"
  nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader
} > "$PHASE_DIR/launch.txt"

sha256sum \
  EVENT_STATE_ROADMAP.md \
  configs/zipcount_v4_distill.yaml \
  src/models/heads.py \
  src/models/losses.py \
  src/models/zipcount_v1.py \
  src/train.py \
  scripts/eval_per_recording.py \
  scripts/compare_runs.py \
  scripts/segment_diagnosis.py \
  scripts/run_phase7_event_state.sh \
  "$TRAIN_MANIFEST" "$VAL_MANIFEST" "$VOX_SEL" \
  > "$HASH_FILE"

for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do
    run_one "$arm" "$seed"
  done
done

{
  echo "=== LOCKED STEP 400: C1 event supervision vs C0 ==="
  "$PY" scripts/compare_runs.py \
    --a results/p7_c0_s1234_step400_voxsel.json results/p7_c0_s2345_step400_voxsel.json results/p7_c0_s3456_step400_voxsel.json \
    --b results/p7_c1_s1234_step400_voxsel.json results/p7_c1_s2345_step400_voxsel.json results/p7_c1_s3456_step400_voxsel.json \
    --name-a C0 --name-b C1 --metric macro_f1
  echo "=== LOCKED STEP 400: C2 structured system vs C1 ==="
  "$PY" scripts/compare_runs.py \
    --a results/p7_c1_s1234_step400_voxsel.json results/p7_c1_s2345_step400_voxsel.json results/p7_c1_s3456_step400_voxsel.json \
    --b results/p7_c2_s1234_step400_voxsel.json results/p7_c2_s2345_step400_voxsel.json results/p7_c2_s3456_step400_voxsel.json \
    --name-a C1 --name-b C2 --metric macro_f1
  echo "=== LOCKED STEP 400: C2 structured system vs C0 ==="
  "$PY" scripts/compare_runs.py \
    --a results/p7_c0_s1234_step400_voxsel.json results/p7_c0_s2345_step400_voxsel.json results/p7_c0_s3456_step400_voxsel.json \
    --b results/p7_c2_s1234_step400_voxsel.json results/p7_c2_s2345_step400_voxsel.json results/p7_c2_s3456_step400_voxsel.json \
    --name-a C0 --name-b C2 --metric macro_f1
} | tee "$COMPARE_FILE"

for arm in "${ARMS[@]}"; do
  PYTHONPATH=. "$PY" scripts/segment_diagnosis.py \
    --manifest "$VOX_SEL" \
    --checkpoint "logs/p7_${arm}_s1234/step${PRIMARY_STEP}.pt" \
    --config "$PHASE_DIR/p7_${arm}_s1234.yaml" \
    --teacher-dir "$TEACHER_DIR" \
    --tolerances-ms "40,80,160,250" \
    > "$PHASE_DIR/p7_${arm}_s1234_step${PRIMARY_STEP}_segments.txt"
done

date --iso-8601=seconds > "$PHASE_DIR/COMPLETE"
echo "Phase 7 complete. Locked comparisons: $COMPARE_FILE"
