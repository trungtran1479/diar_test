#!/bin/bash
# Diagnostic follow-up to run_phase8_finetune.sh — NOT a new adoption gate.
#
# Two new seeds (4567, 5678) trained head-only under the EXACT a0-winner
# config (chain.base_config(), unmodified) so we have 5 seeds total for the
# fine-tune instability question raised by seed 2345's post-finetune drop.
# Same regime as the chain's own a0 arm: frozen backbone, require_backbone_
# complete, reset_rng_after_init, 3000 steps, locked endpoint.
set -e
cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python
TMP=/tmp/claude-1000/-home-edabk-hoangbpm/c22de1cf-7989-4e68-989d-f720938175c6/scratchpad
INIT=logs/r8_v3_diverse/best_macro_f1.pt

need_gb=16
avail_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "$avail_gb" -ge "$need_gb" ] || { echo "FATAL: ${avail_gb}G free" >&2; exit 1; }

run_one () {
  local seed=$1
  local tag="p8a0_pyr_s${seed}"
  local cfg="$TMP/${tag}.yaml"
  PYTHONPATH=. "$PY" - "$cfg" "$seed" "$tag" <<'PYEOF'
import sys
sys.path.insert(0, "scripts")
import yaml
import phase8_chain_driver as chain

cfg_out, seed, tag = sys.argv[1:4]
c = chain.base_config()
chain.finalize_training_block(c, int(seed), tag)
with open(cfg_out, "w") as f:
    yaml.safe_dump(c, f, sort_keys=False)
PYEOF
  echo "=== ${tag} (head-only, a0-winner config, seed=$seed) ==="
  rm -rf "logs/${tag}"
  PYTHONPATH=. "$PY" src/train.py --config "$cfg" --init-from "$INIT" \
      > "logs/${tag}.log" 2>&1
  grep -aE "^\[val@3000\]" "logs/${tag}.log" | sed 's/acc=[^ ]* mae=[^ ]* //'
  find "logs/${tag}" -type f ! -name 'step3000.pt' ! -name 'events*' -delete
}

for seed in 4567 5678; do
  run_one "$seed"
done

echo "=== SCORING new head-only seeds (locked step3000) ==="
SEL="/media/edabk/500GB Hard Disk/data_diar/voxconverse/vox_sel.json"
for seed in 4567 5678; do
  tag="p8a0_pyr_s${seed}"
  PYTHONPATH=. "$PY" scripts/eval_per_recording.py \
      --manifest "$SEL" --checkpoint "logs/${tag}/step3000.pt" \
      --config "$TMP/${tag}.yaml" \
      --out "results/${tag}_voxsel.json"
done
echo "=== HEAD-ONLY EXTENSION DONE ==="
