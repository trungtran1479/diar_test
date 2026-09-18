#!/bin/bash
# Diagnostic follow-up to run_phase8_finetune.sh — NOT a new adoption gate.
#
# Question: seed 2345 dropped sharply after the 400-step full-unfreeze
# finetune at lr=3e-5 (both vs p2_ctrl and vs p6c_tcnft), while every other
# seed was fine or improved. Two things could explain it: (a) seed 2345 is
# simply an unstable draw under this lr, or (b) lr=3e-5 is too aggressive for
# full unfreeze from a head-only mask-restricted checkpoint and would show
# similar instability elsewhere with more seeds.
#
# This sweep runs BOTH controls in one pass, seeded consistently:
#   - lr in {1e-5, 1.5e-5}, ALL 5 seeds (1234, 2345, 3456, 4567, 5678)
#   - lr=3e-5 (matches run_phase8_finetune.sh exactly) for the two NEW seeds
#     only (4567, 5678); 1234/2345/3456 at 3e-5 already exist as p8ft_s*.
#
# Same 400-step locked endpoint and regime as run_phase8_finetune.sh
# (stage3_finetune, all params trainable, KD/T-MSE off) — only training.lr
# and the seed set differ. Requires run_phase8_more_seeds.sh to have produced
# logs/p8a0_pyr_s{4567,5678}/step3000.pt first.
set -e
cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python
STEPS=400
TMP=/tmp/claude-1000/-home-edabk-hoangbpm/c22de1cf-7989-4e68-989d-f720938175c6/scratchpad

need_gb=16
avail_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "$avail_gb" -ge "$need_gb" ] || { echo "FATAL: ${avail_gb}G free" >&2; exit 1; }

lr_tag () {  # 1e-5 -> lr1e5 ; 1.5e-5 -> lr15e5
  case "$1" in
    1e-5) echo lr1e5 ;;
    1.5e-5) echo lr15e5 ;;
    3e-5) echo lr3e5 ;;
    *) echo "unknown lr $1" >&2; exit 1 ;;
  esac
}

run_one () {
  local lr=$1 seed=$2
  local ltag; ltag=$(lr_tag "$lr")
  local tag="p8ft${ltag}_s${seed}"
  local cfg="$TMP/${tag}.yaml"
  local init="logs/p8a0_pyr_s${seed}/step3000.pt"
  [ -f "$init" ] || { echo "FATAL: missing $init" >&2; exit 1; }
  PYTHONPATH=. "$PY" - "$cfg" "$STEPS" "$seed" "$tag" "$lr" <<'PYEOF'
import sys
sys.path.insert(0, "scripts")
import yaml
import phase8_chain_driver as chain

cfg_out, steps, seed, tag, lr = sys.argv[1:6]
c = chain.base_config()
t = c["training"]
t["stage"] = "stage3_finetune"
t["freeze_backbone_all"] = False
t.pop("init_backbone_only", None)
t.pop("require_backbone_complete", None)
t.pop("reset_rng_after_init", None)
t["lr"] = float(lr)
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
  echo "=== ${tag} (finetune lr=${lr} from p8a0_pyr head-only, seed=$seed) ==="
  rm -rf "logs/${tag}"
  PYTHONPATH=. "$PY" src/train.py --config "$cfg" --init-from "$init" \
      > "logs/${tag}.log" 2>&1
  grep -aE "^\[val@400\]" "logs/${tag}.log" | sed 's/acc=[^ ]* mae=[^ ]* //'
  find "logs/${tag}" -type f ! -name 'step400.pt' ! -name 'events*' -delete
}

for seed in 1234 2345 3456 4567 5678; do
  run_one 1e-5 "$seed"
  run_one 1.5e-5 "$seed"
done
for seed in 4567 5678; do
  run_one 3e-5 "$seed"
done

echo "=== LR SWEEP TRAINING DONE — scoring the LOCKED step400 endpoint ==="
SEL="/media/edabk/500GB Hard Disk/data_diar/voxconverse/vox_sel.json"
for seed in 1234 2345 3456 4567 5678; do
  for lr in 1e-5 1.5e-5; do
    ltag=$(lr_tag "$lr")
    tag="p8ft${ltag}_s${seed}"
    PYTHONPATH=. "$PY" scripts/eval_per_recording.py \
        --manifest "$SEL" --checkpoint "logs/${tag}/step400.pt" \
        --config "$TMP/${tag}.yaml" \
        --out "results/${tag}_voxsel.json"
  done
done
for seed in 4567 5678; do
  tag="p8ftlr3e5_s${seed}"
  PYTHONPATH=. "$PY" scripts/eval_per_recording.py \
      --manifest "$SEL" --checkpoint "logs/${tag}/step400.pt" \
      --config "$TMP/${tag}.yaml" \
      --out "results/${tag}_voxsel.json"
done
echo "=== LR SWEEP EVAL DONE ==="
