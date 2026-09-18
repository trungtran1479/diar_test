#!/bin/bash
# Phase 6: GRADUATION — both Phase-5c heads get the identical full-finetune
# treatment, giving a SYMMETRIC architecture comparison after unfreeze plus a
# system comparison against the historical incumbent.
#
# Pre-registered before running:
#   arms            tcnft    — init logs/p5c_tcn192_s{seed}/step3000.pt
#                   deformft — init logs/p5c_deform_s{seed}/step3000.pt
#                   During Phase 5c the backbone was FROZEN, so each init is
#                   exactly the v3 backbone + that seed's head, trained the
#                   same 3000 steps under the same regime. The two lineages
#                   are fully symmetric: same data, same steps, same losses.
#   regime          uniform LR 3e-5, all params trainable, KD off, legacy KL
#                   smoothness 0.02 UNCHANGED (as in every historical run),
#                   boundary T-MSE OFF via its distinct key.
#   PRIMARY         tcnft vs deformft — the architecture question after
#                   unfreeze, with training history CONTROLLED.
#   SECONDARY       tcnft vs p2_ctrl — the deployment/system question against
#                   the incumbent (histories differ: v1->v3 lineage vs 3000
#                   head-only steps — stated, not hidden).
#   endpoint        step 400, LOCKED. vox_sel macro-F1 per recording.
#   verdict         REAL only if all 3 paired deltas share sign AND CI
#                   excludes zero.
set -e
cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python
STEPS=800
TMP=/tmp/claude-1000/-home-edabk-hoangbpm/c22de1cf-7989-4e68-989d-f720938175c6/scratchpad

need_gb=16
avail_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "$avail_gb" -ge "$need_gb" ] || { echo "FATAL: ${avail_gb}G free" >&2; exit 1; }

run_one () {
  local arm=$1 seed=$2
  local tag="${arm}_s${seed}"
  local cfg="$TMP/p6c_${tag}.yaml"
  local init
  if [ "$arm" = "tcnft" ]; then
    init="logs/p5c_tcn192_s${seed}/step3000.pt"
  else
    init="logs/p5c_deform_s${seed}/step3000.pt"
  fi
  [ -f "$init" ] || { echo "FATAL: missing $init" >&2; exit 1; }
  "$PY" - "$cfg" "$arm" "$STEPS" "$seed" "$tag" <<'PYEOF'
import sys, yaml
cfg_out, arm, steps, seed, tag = sys.argv[1:6]
c = yaml.safe_load(open("configs/zipcount_v4_distill.yaml"))
c["distill"]["lambda_kd"] = 0.0
c["loss"]["lambda_boundary_smooth"] = 0.0
if arm == "tcnft":
    c["model"]["head"] = {"type": "tcn_ordinal", "d_model": 192, "dropout": 0.1}
# deformft keeps the incumbent head config
t = c["training"]
t["max_steps"] = int(steps)
t["seed"] = int(seed)
t["eval_interval"] = 200
t["eval_interval_early"] = 200
t["early_phase_steps"] = int(steps)
t["save_every_eval"] = True
t["log_dir"] = f"logs/p6c_{tag}"
yaml.safe_dump(c, open(cfg_out, "w"), sort_keys=False)
PYEOF
  echo "=== p6c_${tag} (full finetune from p5c head, seed=$seed) ==="
  rm -rf "logs/p6c_${tag}"
  PYTHONPATH=. "$PY" src/train.py --config "$cfg" --init-from "$init" \
      > "logs/p6c_${tag}.log" 2>&1
  grep -aE "^\[val@400\]" "logs/p6c_${tag}.log" | sed 's/acc=[^ ]* mae=[^ ]* //'
  find "logs/p6c_${tag}" -type f ! -name 'step400.pt' ! -name 'events*' -delete
}

for seed in 1234 2345 3456; do
  run_one deformft "$seed"
  run_one tcnft "$seed"
done

echo "=== PHASE 6 TRAINING DONE — scoring the LOCKED step400 endpoint ==="
SEL="/media/edabk/500GB Hard Disk/data_diar/voxconverse/vox_sel.json"
for seed in 1234 2345 3456; do
  for arm in deformft tcnft; do
    PYTHONPATH=. "$PY" scripts/eval_per_recording.py \
        --manifest "$SEL" --checkpoint "logs/p6c_${arm}_s${seed}/step400.pt" \
        --config "$TMP/p6c_${arm}_s${seed}.yaml" \
        --out "results/p6c_${arm}_s${seed}_voxsel.json"
  done
done

echo "=== PRIMARY: tcnft vs deformft (symmetric lineages, paired) ==="
"$PY" scripts/compare_runs.py \
    --a results/p6c_deformft_s1234_voxsel.json results/p6c_deformft_s2345_voxsel.json results/p6c_deformft_s3456_voxsel.json \
    --b results/p6c_tcnft_s1234_voxsel.json results/p6c_tcnft_s2345_voxsel.json results/p6c_tcnft_s3456_voxsel.json \
    --name-a "deform-ft" --name-b "tcn-ft" --metric macro_f1

echo "=== SECONDARY: tcnft vs historical incumbent p2_ctrl (system) ==="
"$PY" scripts/compare_runs.py \
    --a results/p2_ctrl_s1234_voxsel.json results/p2_ctrl_s2345_voxsel.json results/p2_ctrl_s3456_voxsel.json \
    --b results/p6c_tcnft_s1234_voxsel.json results/p6c_tcnft_s2345_voxsel.json results/p6c_tcnft_s3456_voxsel.json \
    --name-a "p2_ctrl" --name-b "tcn-ft" --metric macro_f1

echo "=== MECHANISM (seed 1234, both arms) ==="
for arm in deformft tcnft; do
  echo "--- $arm ---"
  PYTHONPATH=. "$PY" scripts/segment_diagnosis.py --manifest "$SEL" \
      --checkpoint "logs/p6c_${arm}_s1234/step400.pt" --config "$TMP/p6c_${arm}_s1234.yaml" \
      --teacher-dir "/media/edabk/500GB Hard Disk/data_diar/voxconverse/diarizen_on_test" \
      | grep -aA9 "===== zipcount"
done
