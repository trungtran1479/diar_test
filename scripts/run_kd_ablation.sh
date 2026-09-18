#!/bin/bash
# Controlled KD ablation. Every arm starts from the SAME v3 checkpoint with the
# SAME seed, so differences are attributable to the KD setting and not to batch
# order or extra optimisation.
#
# Arms:
#   lam0      lambda_kd=0            -> control: how much is just more training?
#   lam025_T1 lambda_kd=0.25, T=1    -> gentle KD, no temperature games
#   lam025_T2 lambda_kd=0.25, T=2    -> gentle KD, temperature now applied to
#                                       BOTH sides (teacher softened q^(1/T))
#   lam1_T1   lambda_kd=1.0,  T=1    -> strong KD without the sharpening bug
#
# v4 (lambda=1, T=2 with the OLD one-sided temperature) is the existing run to
# compare against: logs/r9_v4_distill.
set -e
cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python
INIT=logs/r8_v3_diverse/best_macro_f1.pt
STEPS=${STEPS:-1600}
SEED=${SEED:-1234}
TMP=/tmp/claude-1000/-home-edabk-hoangbpm/c22de1cf-7989-4e68-989d-f720938175c6/scratchpad

run_arm () {
  local name=$1 lam=$2 temp=$3
  local cfg="$TMP/kd_${name}.yaml"
  "$PY" - "$cfg" "$lam" "$temp" "$name" "$STEPS" "$SEED" <<'PYEOF'
import sys, yaml
cfg_out, lam, temp, name, steps, seed = sys.argv[1:7]
c = yaml.safe_load(open("configs/zipcount_v4_distill.yaml"))
c["distill"]["lambda_kd"] = float(lam)
c["distill"]["temperature"] = float(temp)
c["training"]["max_steps"] = int(steps)
c["training"]["seed"] = int(seed)
c["training"]["eval_interval"] = 200
c["training"]["eval_interval_early"] = 200
c["training"]["early_phase_steps"] = int(steps)
c["training"]["log_dir"] = f"logs/kd_abl_{name}"
yaml.safe_dump(c, open(cfg_out, "w"), sort_keys=False)
print(f"  cfg {cfg_out}: lambda_kd={lam} T={temp} steps={steps} seed={seed}")
PYEOF
  echo "=== arm $name (lambda=$lam, T=$temp) ==="
  rm -rf "logs/kd_abl_${name}"
  PYTHONPATH=. "$PY" src/train.py --config "$cfg" --init-from "$INIT" \
      > "logs/kd_abl_${name}.log" 2>&1
  echo -n "  best: "
  grep -aoE "macroF1=[0-9.]+" "logs/kd_abl_${name}.log" | sort -t= -k2 -rn | head -1
}

run_arm lam0      0.0  1.0
run_arm lam025_T1 0.25 1.0
run_arm lam025_T2 0.25 2.0
run_arm lam1_T1   1.0  1.0

echo "=== ALL ARMS DONE ==="
for a in lam0 lam025_T1 lam025_T2 lam1_T1; do
  echo "--- $a ---"; grep -aE "^\[val@" "logs/kd_abl_${a}.log" | sed 's/acc=[^ ]* mae=[^ ]* //'
done
