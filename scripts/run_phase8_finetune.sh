#!/bin/bash
# Phase 8 graduation: full-finetune the ONLY chain arm the strict ladder
# adopted — a0's pyramid-minimal base (mask [1,1,1,0,0,0], gate_mode global,
# softmax ordinal, no stage2, legacy v4 loss).  Every later mutation tried in
# the chain (framewise gate a1, final residual a2, SORD/cumulative structured
# loss phase b) failed its own adoption bar or bridge and was rejected before
# this point, so the head-only checkpoint to fine-tune is exactly
# logs/p8a0_pyr_s{seed}/step3000.pt.
#
# Regime is IDENTICAL to Phase 6 (run_phase6.sh): uniform LR 3e-5 (the
# pyramid config's own default), all params trainable, KD off, boundary
# T-MSE off, legacy KL smoothness 0.02 unchanged.  Endpoint step 400, LOCKED.
#
#   PRIMARY   p8ft vs p2_ctrl   — system question: does the new architecture
#             (pyramid + pre012 mask), after the SAME unfreeze regime, beat
#             the historical incumbent on vox_sel.
#   SECONDARY p8ft vs p6c_tcnft — architecture question under a symmetric
#             finetune history (both are "3000 head-only steps then 400
#             finetune steps"), TCN-concat family vs pyramid family.
#
# Verdict: REAL only if all 3 paired deltas share sign AND CI excludes zero.
set -e
cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python
STEPS=400
TMP=/tmp/claude-1000/-home-edabk-hoangbpm/c22de1cf-7989-4e68-989d-f720938175c6/scratchpad

need_gb=16
avail_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "$avail_gb" -ge "$need_gb" ] || { echo "FATAL: ${avail_gb}G free" >&2; exit 1; }

run_one () {
  local seed=$1
  local tag="p8ft_s${seed}"
  local cfg="$TMP/${tag}.yaml"
  local init="logs/p8a0_pyr_s${seed}/step3000.pt"
  [ -f "$init" ] || { echo "FATAL: missing $init" >&2; exit 1; }
  PYTHONPATH=. "$PY" - "$cfg" "$STEPS" "$seed" "$tag" <<'PYEOF'
import sys
sys.path.insert(0, "scripts")
import yaml
import phase8_chain_driver as chain

cfg_out, steps, seed, tag = sys.argv[1:5]
c = chain.base_config()
# base_config() already carries the ONLY adopted mutation (none beyond the
# mask baked into rev-7) and zeroes KD/boundary-smooth/event/raw-anchor.
# Full-finetune regime, mirroring run_phase6.sh on top of the same file:
t = c["training"]
t["stage"] = "stage3_finetune"
t["freeze_backbone_all"] = False
t.pop("init_backbone_only", None)
t.pop("require_backbone_complete", None)
t.pop("reset_rng_after_init", None)
t["max_steps"] = int(steps)
t["seed"] = int(seed)
t["eval_interval"] = 200
t["eval_interval_early"] = 200
t["early_phase_steps"] = int(steps)
t["save_every_eval"] = True
t["snapshot_weights_only"] = True
t["save_best"] = False
t["save_last"] = False
t["log_dir"] = f"logs/{tag}"
c["model"]["encoder"]["freeze_backbone"] = False
with open(cfg_out, "w") as f:
    yaml.safe_dump(c, f, sort_keys=False)
PYEOF
  echo "=== ${tag} (full finetune from p8a0_pyr head-only, seed=$seed) ==="
  rm -rf "logs/${tag}"
  PYTHONPATH=. "$PY" src/train.py --config "$cfg" --init-from "$init" \
      > "logs/${tag}.log" 2>&1
  grep -aE "^\[val@400\]" "logs/${tag}.log" | sed 's/acc=[^ ]* mae=[^ ]* //'
  find "logs/${tag}" -type f ! -name 'step400.pt' ! -name 'events*' -delete
}

for seed in 1234 2345 3456; do
  run_one "$seed"
done

echo "=== PHASE 8 FINETUNE TRAINING DONE — scoring the LOCKED step400 endpoint ==="
SEL="/media/edabk/500GB Hard Disk/data_diar/voxconverse/vox_sel.json"
for seed in 1234 2345 3456; do
  PYTHONPATH=. "$PY" scripts/eval_per_recording.py \
      --manifest "$SEL" --checkpoint "logs/p8ft_s${seed}/step400.pt" \
      --config "$TMP/p8ft_s${seed}.yaml" \
      --out "results/p8ft_s${seed}_voxsel.json"
done

echo "=== PRIMARY: p2_ctrl vs p8ft (system, paired) ==="
"$PY" scripts/compare_runs.py \
    --a results/p2_ctrl_s1234_voxsel.json results/p2_ctrl_s2345_voxsel.json results/p2_ctrl_s3456_voxsel.json \
    --b results/p8ft_s1234_voxsel.json results/p8ft_s2345_voxsel.json results/p8ft_s3456_voxsel.json \
    --name-a "p2_ctrl" --name-b "p8ft" --metric macro_f1

echo "=== SECONDARY: p6c_tcnft vs p8ft (architecture, symmetric finetune history) ==="
"$PY" scripts/compare_runs.py \
    --a results/p6c_tcnft_s1234_voxsel.json results/p6c_tcnft_s2345_voxsel.json results/p6c_tcnft_s3456_voxsel.json \
    --b results/p8ft_s1234_voxsel.json results/p8ft_s2345_voxsel.json results/p8ft_s3456_voxsel.json \
    --name-a "p6c_tcnft" --name-b "p8ft" --metric macro_f1
