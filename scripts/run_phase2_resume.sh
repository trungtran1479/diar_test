#!/bin/bash
# Resume Phase 2 at seed 3456. Seeds 1234/2345 completed and their LOCKED
# step400.pt endpoints are kept; everything else from those runs was deleted to
# free disk. The original run died mid-ctrl_s3456 because / hit 100% and
# torch.save wrote a truncated checkpoint, so seed 3456 restarts from scratch
# for BOTH arms — the pairing only holds if both arms of a seed are trained
# under identical conditions.
#
# Design is unchanged from run_phase2.sh and stays pre-registered:
#   endpoint step 400 LOCKED, primary vox_sel macro-F1, arms lambda=0 vs
#   lambda=0.25 T=2, paired seeds.
set -e
cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python
INIT=logs/r8_v3_diverse/best_macro_f1.pt
STEPS=800
TMP=/tmp/claude-1000/-home-edabk-hoangbpm/c22de1cf-7989-4e68-989d-f720938175c6/scratchpad

# a truncated checkpoint is worse than a crash: it looks loadable. Refuse to
# start unless there is room for every checkpoint this script will write.
need_gb=16
avail_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$avail_gb" -lt "$need_gb" ]; then
  echo "FATAL: only ${avail_gb}G free on /, need ${need_gb}G" >&2
  exit 1
fi

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
  # keep only the locked endpoint; the rest is 3GB of disk we do not have
  find "logs/p2_${tag}" -type f ! -name 'step400.pt' ! -name 'events*' -delete
}

run_one ctrl 0.0  1.0 3456
run_one kd   0.25 2.0 3456

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
