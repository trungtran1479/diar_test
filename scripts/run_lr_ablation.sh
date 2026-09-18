#!/bin/bash
# Phase 1: isolate the DRIFT cause. Every arm is lambda_kd=0, so nothing here is
# about distillation — only about how continuing to train damages AMI.
#
#   uniform    backbone 3e-5, head 3e-5   (the existing setting)
#   bb_div10   backbone 3e-6, head 3e-5
#   bb_div30   backbone 1e-6, head 3e-5
#   head_only  backbone frozen, head 3e-5 (cleanest anchor)
#
# Head LR is deliberately left at 3e-5 in every arm: raising it would change two
# variables at once. Scheduler, weight decay, seed and batch order are identical.
# Checkpoints are snapshotted at every eval so "just stop earlier" can be tested
# as an alternative explanation to optimizer surgery.
set -e
cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python
INIT=logs/r8_v3_diverse/best_macro_f1.pt
STEPS=${STEPS:-1600}
SEED=${SEED:-1234}
TMP=/tmp/claude-1000/-home-edabk-hoangbpm/c22de1cf-7989-4e68-989d-f720938175c6/scratchpad

run_arm () {
  local name=$1 bblr=$2 freeze=$3
  local cfg="$TMP/lr_${name}.yaml"
  "$PY" - "$cfg" "$bblr" "$freeze" "$name" "$STEPS" "$SEED" <<'PYEOF'
import sys, yaml
cfg_out, bblr, freeze, name, steps, seed = sys.argv[1:7]
c = yaml.safe_load(open("configs/zipcount_v4_distill.yaml"))
c["distill"]["lambda_kd"] = 0.0            # phase 1 is about drift, not KD
c["training"]["max_steps"] = int(steps)
c["training"]["seed"] = int(seed)
c["training"]["eval_interval"] = 200
c["training"]["eval_interval_early"] = 200
c["training"]["early_phase_steps"] = int(steps)
c["training"]["save_every_eval"] = True
c["training"]["head_lr"] = 3.0e-5
if freeze == "1":
    c["training"]["freeze_backbone_all"] = True
else:
    c["training"]["backbone_lr"] = float(bblr)
c["training"]["log_dir"] = f"logs/lr_abl_{name}"
yaml.safe_dump(c, open(cfg_out, "w"), sort_keys=False)
PYEOF
  echo "=== arm $name (backbone_lr=$bblr freeze=$freeze) ==="
  rm -rf "logs/lr_abl_${name}"
  PYTHONPATH=. "$PY" src/train.py --config "$cfg" --init-from "$INIT" \
      > "logs/lr_abl_${name}.log" 2>&1
  grep -aE "Differential LR|freeze_backbone_all|Number of trainable" "logs/lr_abl_${name}.log" | head -2
  echo -n "  best AMI-dev macroF1: "
  grep -aoE "macroF1=[0-9.]+" "logs/lr_abl_${name}.log" | sort -t= -k2 -rn | head -1
}

run_arm uniform   3.0e-5 0
run_arm bb_div10  3.0e-6 0
run_arm bb_div30  1.0e-6 0
run_arm head_only 0      1

echo "=== PHASE 1 DONE ==="
for a in uniform bb_div10 bb_div30 head_only; do
  echo "--- $a ---"; grep -aE "^\[val@" "logs/lr_abl_${a}.log" | sed 's/acc=[^ ]* mae=[^ ]* //'
done
