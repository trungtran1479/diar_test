#!/bin/bash
# HISTORICAL NOTE (post-hoc): the run recorded in logs/p4_* was CONFOUNDED —
# at the time, the boundary T-MSE shared the `lambda_smooth` key with the
# legacy symmetric-KL smoothness in losses.py, so the arm changed legacy KL
# 0.02->0.15 AND added T-MSE@0.15 in one move. Its +0.0007 only means "more
# total smoothing ~ harmless". The keys below have been migrated to the new
# distinct names, so a re-run today WOULD be the clean ablation the header
# describes; the p4_* artifacts remain from the confounded version.
# Phase 4: boundary-aware temporal smoothing — the smallest intervention aimed
# at the MEASURED failure mode.
#
# Pre-registered before running:
#   diagnosis       (segment_diagnosis.py, vox_sel) ZipCount already OUT-HEARS
#                   the teacher on overlap: onset recall 0.843 vs 0.707, missed
#                   segments 13.6% vs 25.8%. The deficit is structural:
#                   fragmentation 2.59x vs 1.27x, boundary F1 0.329 vs 0.540.
#                   A crude 1 s median filter buys +0.018 macro-F1, so the
#                   flicker headroom is real; a LEARNED, boundary-gated
#                   smoothness should reach further (filters cannot fix the
#                   35% partially-covered segments).
#   arm             lambda_smooth=0.15, tau=4, guard=±2 frames (MS-TCN's
#                   T-MSE numbers; guard is ours). KD off. Everything else
#                   identical to Phase 2's control regime.
#   control         the EXISTING logs/p2_ctrl_s{seed}/step400.pt — same seeds,
#                   same batch order, so the pairing is exact.
#   endpoint        step 400, LOCKED (same as Phase 2/3).
#   primary metric  vox_sel macro-F1 per recording, paired bootstrap.
#   secondary       fragmentation rate + boundary F1 from segment_diagnosis
#                   (the mechanism check: did it actually de-flicker?).
#   verdict         REAL only if all 3 seeds positive AND recording CI
#                   excludes zero. Seed noise floor is ~0.02 pooled macro-F1.
set -e
cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python
INIT=logs/r8_v3_diverse/best_macro_f1.pt
STEPS=800
TMP=/tmp/claude-1000/-home-edabk-hoangbpm/c22de1cf-7989-4e68-989d-f720938175c6/scratchpad

need_gb=16
avail_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "$avail_gb" -ge "$need_gb" ] || { echo "FATAL: ${avail_gb}G free, need ${need_gb}G" >&2; exit 1; }
for seed in 1234 2345 3456; do
  [ -f "logs/p2_ctrl_s${seed}/step400.pt" ] || { echo "FATAL: missing paired control s${seed}" >&2; exit 1; }
done

run_one () {
  local seed=$1
  local tag="smooth_s${seed}"
  local cfg="$TMP/p4_${tag}.yaml"
  "$PY" - "$cfg" "$STEPS" "$seed" "$tag" <<'PYEOF'
import sys, yaml
cfg_out, steps, seed, tag = sys.argv[1:5]
c = yaml.safe_load(open("configs/zipcount_v4_distill.yaml"))
c["distill"]["lambda_kd"] = 0.0
c["loss"]["lambda_boundary_smooth"] = 0.15
c["loss"]["boundary_smooth_tau"] = 4.0
c["loss"]["boundary_smooth_guard"] = 2
c["training"]["max_steps"] = int(steps)
c["training"]["seed"] = int(seed)
c["training"]["eval_interval"] = 200
c["training"]["eval_interval_early"] = 200
c["training"]["early_phase_steps"] = int(steps)
c["training"]["save_every_eval"] = True
c["training"]["log_dir"] = f"logs/p4_{tag}"
yaml.safe_dump(c, open(cfg_out, "w"), sort_keys=False)
PYEOF
  echo "=== p4_${tag} (lambda_smooth=0.15 seed=$seed) ==="
  rm -rf "logs/p4_${tag}"
  PYTHONPATH=. "$PY" src/train.py --config "$cfg" --init-from "$INIT" \
      > "logs/p4_${tag}.log" 2>&1
  grep -aE "^\[val@400\]" "logs/p4_${tag}.log" | sed 's/acc=[^ ]* mae=[^ ]* //'
  find "logs/p4_${tag}" -type f ! -name 'step400.pt' ! -name 'events*' -delete
}

for seed in 1234 2345 3456; do run_one "$seed"; done

echo "=== PHASE 4 TRAINING DONE — scoring the LOCKED step400 endpoint ==="
SEL="/media/edabk/500GB Hard Disk/data_diar/voxconverse/vox_sel.json"
for seed in 1234 2345 3456; do
  PYTHONPATH=. "$PY" scripts/eval_per_recording.py \
      --manifest "$SEL" --checkpoint "logs/p4_smooth_s${seed}/step400.pt" \
      --config configs/zipcount_v4_distill.yaml \
      --out "results/p4_smooth_s${seed}_voxsel.json"
done

echo "=== PRIMARY: paired comparison vs lambda=0 control ==="
"$PY" scripts/compare_runs.py \
    --a results/p2_ctrl_s1234_voxsel.json results/p2_ctrl_s2345_voxsel.json results/p2_ctrl_s3456_voxsel.json \
    --b results/p4_smooth_s1234_voxsel.json results/p4_smooth_s2345_voxsel.json results/p4_smooth_s3456_voxsel.json \
    --name-a "ctrl" --name-b "smooth .15" --metric macro_f1

echo "=== MECHANISM CHECK: did it de-flicker? (seed 1234) ==="
PYTHONPATH=. "$PY" scripts/segment_diagnosis.py --manifest "$SEL" \
    --checkpoint logs/p4_smooth_s1234/step400.pt \
    --teacher-dir "/media/edabk/500GB Hard Disk/data_diar/voxconverse/diarizen_on_test" \
    | grep -aA8 "===== zipcount"
