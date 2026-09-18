#!/bin/bash
# Phase 3: does distilling on RIGHT-BUT-UNCERTAIN frames do what "agree" could not?
#
# Pre-registered before running:
#   motivation      Phase 2 (3 paired seeds) put the `agree` mask at +0.0007,
#                   CI [-0.0001,+0.0015] — not established. probe_kd_mask.py
#                   explains why: `agree` keeps 89% of frames at mean max-prob
#                   0.9312 with only 0.0614 on the runner-up, and is biased
#                   toward the class the student already handles (95.0% of
#                   class-1 vs 48.3% of class-3). That KL is a reweighted hard
#                   label. `agree & maxp<0.9` keeps 19.2% and inverts the bias
#                   (12.5% of class-1 vs 43.9%/35.2% of class-2/3).
#
#   PRIMARY arm     mask_mode=agree_uncertain, lambda=0.25, T=2 — identical to
#                   the Phase 2 KD arm in EVERY other respect, so the contrast
#                   isolates the mask and nothing else.
#   SECONDARY arm   same mask at lambda=1.0. EXPLORATORY: reported with the
#                   multiplicity stated, never promoted to the headline. Two
#                   arms scored on the same metric means picking the winner
#                   post hoc would inflate the false-positive rate.
#
#   CONTROL         the EXISTING logs/p2_ctrl_s{seed}/step400.pt. Same seeds,
#                   same batch order, same everything but lambda — reusing them
#                   keeps the pairing exact and costs three fewer runs.
#   endpoint        step 400, LOCKED, same as Phase 2.
#   primary metric  VoxConverse vox_sel macro-F1, per recording.
#   verdict rule    REAL only if the paired delta is positive across all three
#                   seeds AND the recording bootstrap CI excludes zero. The
#                   Phase 2 noise floor is ~0.02 macro-F1 of within-arm seed
#                   spread, so a single-seed number proves nothing here.
#   safety          AMI-dev must not degrade beyond seed variability.
#
# vox_lock and msdwild_manyval_LOCKED.json stay SEALED. They are spent once, on
# a result that has already passed the primary endpoint.
set -e
cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python
INIT=logs/r8_v3_diverse/best_macro_f1.pt
STEPS=800
TMP=/tmp/claude-1000/-home-edabk-hoangbpm/c22de1cf-7989-4e68-989d-f720938175c6/scratchpad

need_gb=16
avail_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$avail_gb" -lt "$need_gb" ]; then
  echo "FATAL: only ${avail_gb}G free on /, need ${need_gb}G" >&2
  exit 1
fi

for seed in 1234 2345 3456; do
  if [ ! -f "logs/p2_ctrl_s${seed}/step400.pt" ]; then
    echo "FATAL: missing paired control logs/p2_ctrl_s${seed}/step400.pt" >&2
    exit 1
  fi
done

run_one () {
  local tag=$1 lam=$2 seed=$3
  local cfg="$TMP/p3_${tag}_s${seed}.yaml"
  "$PY" - "$cfg" "$lam" "${tag}_s${seed}" "$STEPS" "$seed" <<'PYEOF'
import sys, yaml
cfg_out, lam, tag, steps, seed = sys.argv[1:6]
c = yaml.safe_load(open("configs/zipcount_v4_distill.yaml"))
c["distill"]["lambda_kd"] = float(lam)
c["distill"]["temperature"] = 2.0
c["distill"]["mask_mode"] = "agree_uncertain"
c["distill"]["uncertain_max_prob"] = 0.9
c["training"]["max_steps"] = int(steps)
c["training"]["seed"] = int(seed)
c["training"]["eval_interval"] = 200
c["training"]["eval_interval_early"] = 200
c["training"]["early_phase_steps"] = int(steps)
c["training"]["save_every_eval"] = True
c["training"]["log_dir"] = f"logs/p3_{tag}"
yaml.safe_dump(c, open(cfg_out, "w"), sort_keys=False)
PYEOF
  echo "=== p3_${tag}_s${seed} (mask=agree_uncertain lambda=$lam T=2 seed=$seed) ==="
  rm -rf "logs/p3_${tag}_s${seed}"
  PYTHONPATH=. "$PY" src/train.py --config "$cfg" --init-from "$INIT" \
      > "logs/p3_${tag}_s${seed}.log" 2>&1
  grep -aE "^\[val@400\]" "logs/p3_${tag}_s${seed}.log" | sed 's/acc=[^ ]* mae=[^ ]* //'
  find "logs/p3_${tag}_s${seed}" -type f ! -name 'step400.pt' ! -name 'events*' -delete
}

for seed in 1234 2345 3456; do
  run_one unc025 0.25 "$seed"   # PRIMARY: mask swap only
  run_one unc100 1.00 "$seed"   # SECONDARY (exploratory)
done

echo "=== PHASE 3 TRAINING DONE — scoring the LOCKED step400 endpoint ==="
SEL="/media/edabk/500GB Hard Disk/data_diar/voxconverse/vox_sel.json"
mkdir -p results
for seed in 1234 2345 3456; do
  for tag in unc025 unc100; do
    PYTHONPATH=. "$PY" scripts/eval_per_recording.py \
        --manifest "$SEL" --checkpoint "logs/p3_${tag}_s${seed}/step400.pt" \
        --config configs/zipcount_v4_distill.yaml \
        --out "results/p3_${tag}_s${seed}_voxsel.json"
  done
done

C="results/p2_ctrl_s1234_voxsel.json results/p2_ctrl_s2345_voxsel.json results/p2_ctrl_s3456_voxsel.json"
echo ""
echo "############ PRIMARY: agree_uncertain lambda=0.25 vs control ############"
"$PY" scripts/compare_runs.py --a $C \
    --b results/p3_unc025_s1234_voxsel.json results/p3_unc025_s2345_voxsel.json results/p3_unc025_s3456_voxsel.json \
    --name-a "lambda=0" --name-b "unc lambda=.25" --metric macro_f1

echo ""
echo "###### SECONDARY (exploratory, 2 arms on one metric): lambda=1.0 ######"
"$PY" scripts/compare_runs.py --a $C \
    --b results/p3_unc100_s1234_voxsel.json results/p3_unc100_s2345_voxsel.json results/p3_unc100_s3456_voxsel.json \
    --name-a "lambda=0" --name-b "unc lambda=1.0" --metric macro_f1

echo ""
echo "###### class-2 F1 (secondary endpoint), primary arm ######"
"$PY" scripts/compare_runs.py --a $C \
    --b results/p3_unc025_s1234_voxsel.json results/p3_unc025_s2345_voxsel.json results/p3_unc025_s3456_voxsel.json \
    --name-a "lambda=0" --name-b "unc lambda=.25" --metric f1_2
