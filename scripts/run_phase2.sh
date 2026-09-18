#!/bin/bash
# Phase 2: does KD add anything, once the optimizer regime is fixed?
#
# Pre-registered before running:
#   regime          uniform LR 3e-5 (Phase 1 selected it under the current
#                   development criterion), early stop
#   PRIMARY endpoint  step 400, LOCKED — the same step for every arm and every
#                   seed. Picking each run's own best step would bias whichever
#                   arm got a lucky eval.
#   primary metric  VoxConverse vox_sel macro-F1
#   secondary       class-2 F1, OSD F1
#   safety          AMI must not degrade beyond recording/seed variability
#   arms            lambda_kd=0  vs  lambda_kd=0.25, T=2 (teacher-softened)
#   seeds           3 PAIRED seeds: for a given seed both arms see the same
#                   batch order and augmentation draws, so the per-seed delta
#                   is a within-pair difference, not two unpaired means.
#
# These are three FINE-TUNING seeds from one v3 checkpoint, not three
# end-to-end training seeds — report it that way.
set -e
cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python
INIT=logs/r8_v3_diverse/best_macro_f1.pt
STEPS=800          # run past the endpoint to see the curve, but compare at 400
TMP=/tmp/claude-1000/-home-edabk-hoangbpm/c22de1cf-7989-4e68-989d-f720938175c6/scratchpad

run_one () {
  local arm=$1 lam=$2 temp=$3 seed=$4
  local tag="${arm}_s${seed}"
  local cfg="$TMP/p2_${tag}.yaml"
  "$PY" - "$cfg" "$lam" "$temp" "$tag" "$STEPS" "$seed" <<'PYEOF'
import sys, yaml
cfg_out, lam, temp, tag, steps, seed = sys.argv[1:7]
c = yaml.safe_load(open("configs/zipcount_v4_distill.yaml"))
c["distill"]["lambda_kd"] = float(lam)
c["distill"]["temperature"] = float(temp)
c["training"]["max_steps"] = int(steps)
c["training"]["seed"] = int(seed)
c["training"]["eval_interval"] = 200
c["training"]["eval_interval_early"] = 200
c["training"]["early_phase_steps"] = int(steps)
c["training"]["save_every_eval"] = True
c["training"]["log_dir"] = f"logs/p2_{tag}"
yaml.safe_dump(c, open(cfg_out, "w"), sort_keys=False)
PYEOF
  echo "=== $tag (lambda=$lam T=$temp seed=$seed) ==="
  rm -rf "logs/p2_${tag}"
  PYTHONPATH=. "$PY" src/train.py --config "$cfg" --init-from "$INIT" \
      > "logs/p2_${tag}.log" 2>&1
  grep -aE "^\[val@400\]" "logs/p2_${tag}.log" | sed 's/acc=[^ ]* mae=[^ ]* //'
}

for seed in 1234 2345 3456; do
  run_one ctrl 0.0  1.0 "$seed"     # control
  run_one kd   0.25 2.0 "$seed"     # KD, teacher softened by the same T
done

echo "=== PHASE 2 TRAINING DONE — scoring the LOCKED step400 endpoint ==="
SEL="/media/edabk/500GB Hard Disk/data_diar/voxconverse/vox_sel.json"
mkdir -p results
for seed in 1234 2345 3456; do
  for arm in ctrl kd; do
    PYTHONPATH=. "$PY" scripts/eval_per_recording.py \
        --manifest "$SEL" --checkpoint "logs/p2_${arm}_s${seed}/step400.pt" \
        --config configs/zipcount_v4_distill.yaml \
        --out "results/p2_${arm}_s${seed}_voxsel.json"
  done
done

echo "=== PAIRED COMPARISON ==="
"$PY" scripts/compare_runs.py \
    --a results/p2_ctrl_s1234_voxsel.json results/p2_ctrl_s2345_voxsel.json results/p2_ctrl_s3456_voxsel.json \
    --b results/p2_kd_s1234_voxsel.json   results/p2_kd_s2345_voxsel.json   results/p2_kd_s3456_voxsel.json \
    --name-a "lambda=0" --name-b "lambda=.25 T=2" --metric macro_f1
