#!/bin/bash
# Phase 5-CLEAN: HEAD ARCHITECTURE — deformable vs dual-dilation TCN, re-run.
#
# WHY A RE-RUN: the first Phase 5 (tags p5_*) accidentally trained BOTH arms
# with the boundary-aware T-MSE active at 0.02 because it shared the
# `lambda_smooth` config key with the legacy KL smoothness. Internally fair
# (both arms identical) but it narrowed the claim to "wins under
# double-smoothing". This run pins the loss regime to EXACTLY the historical
# one (legacy KL 0.02, boundary T-MSE OFF via the new distinct key) so the
# architecture claim is unconditional.
#
# Pre-registered before running:
#   question       Does a decoder with a segment-structure bias beat adaptive
#                  sampling, holding everything else fixed? Diagnosis says the
#                  deficit is structural (fragmentation 2.59x teacher, boundary
#                  F1 0.329 vs 0.540) while overlap recall already beats the
#                  teacher — deformable answers "where to look", not "how long
#                  to hold a decision".
#   protocol       backbone = v3, FROZEN (freeze_backbone_all). Both arms load
#                  ONLY backbone weights (init_backbone_only) so both heads
#                  start RANDOM — loading the incumbent's pretrained head would
#                  measure pretraining, not architecture. Head-only numbers sit
#                  below full-finetune numbers (Phase 1: 0.548 vs 0.562); the
#                  QUESTION here is the ranking between architectures, not the
#                  absolute level. The winner graduates to a full finetune.
#   arms           deform  — incumbent DeformableCountHead, re-initialised
#                  tcn192  — TCNCountHead d_model=192, 1.28M params vs 1.32M:
#                            PARAM-MATCHED, the primary contrast
#                  tcn256  — d_model=256, 2.10M params: capacity probe,
#                            EXPLORATORY (1 seed), never the headline
#   seeds          1234/2345/3456 paired (same batch order per seed)
#   endpoint       step 3000, LOCKED for every arm/seed; curve logged @300
#   primary        vox_sel macro-F1, per-recording paired bootstrap
#   mechanism      segment_diagnosis fragmentation + boundary F1 (did the
#                  structure actually improve, or just the pooled number?)
#   verdict        REAL only if all 3 paired deltas share sign AND the
#                  recording CI excludes zero.
set -e
cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python
INIT=logs/r8_v3_diverse/best_macro_f1.pt
STEPS=3000
TMP=/tmp/claude-1000/-home-edabk-hoangbpm/c22de1cf-7989-4e68-989d-f720938175c6/scratchpad

need_gb=16
avail_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "$avail_gb" -ge "$need_gb" ] || { echo "FATAL: ${avail_gb}G free" >&2; exit 1; }

run_one () {
  local arm=$1 seed=$2
  local tag="${arm}_s${seed}"
  local cfg="$TMP/p5c_${tag}.yaml"
  "$PY" - "$cfg" "$arm" "$STEPS" "$seed" "$tag" <<'PYEOF'
import sys, yaml
cfg_out, arm, steps, seed, tag = sys.argv[1:6]
c = yaml.safe_load(open("configs/zipcount_v4_distill.yaml"))
c["distill"]["lambda_kd"] = 0.0
c["loss"]["lambda_boundary_smooth"] = 0.0   # legacy lambda_smooth stays 0.02 as in every prior run
if arm == "tcn192":
    c["model"]["head"] = {"type": "tcn_ordinal", "d_model": 192, "dropout": 0.1}
elif arm == "tcn256":
    c["model"]["head"] = {"type": "tcn_ordinal", "d_model": 256, "dropout": 0.1}
# arm == "deform": keep the incumbent head config untouched
t = c["training"]
t["max_steps"] = int(steps)
t["seed"] = int(seed)
t["freeze_backbone_all"] = True
t["init_backbone_only"] = True
t["lr"] = 1e-3                      # head from scratch; 3e-5 is a finetune LR
t.pop("backbone_lr", None); t.pop("head_lr", None)
t["eval_interval"] = 300
t["eval_interval_early"] = 300
t["early_phase_steps"] = int(steps)
t["save_every_eval"] = True
t["log_dir"] = f"logs/p5c_{tag}"
yaml.safe_dump(c, open(cfg_out, "w"), sort_keys=False)
PYEOF
  echo "=== p5c_${tag} ==="
  rm -rf "logs/p5c_${tag}"
  PYTHONPATH=. "$PY" src/train.py --config "$cfg" --init-from "$INIT" \
      > "logs/p5c_${tag}.log" 2>&1
  grep -aE "^\[val@(3000|1500)\]" "logs/p5c_${tag}.log" | sed 's/acc=[^ ]* mae=[^ ]* //'
  find "logs/p5c_${tag}" -type f ! -name "step${STEPS}.pt" ! -name 'events*' -delete
}

for seed in 1234 2345 3456; do
  run_one deform "$seed"
  run_one tcn192 "$seed"
done
run_one tcn256 1234        # exploratory capacity probe, 1 seed

echo "=== PHASE 5 TRAINING DONE — scoring the LOCKED step${STEPS} endpoint ==="
SEL="/media/edabk/500GB Hard Disk/data_diar/voxconverse/vox_sel.json"
score () {
  local tag=$1 arm=$2
  PYTHONPATH=. "$PY" scripts/eval_per_recording.py \
      --manifest "$SEL" --checkpoint "logs/p5c_${tag}/step${STEPS}.pt" \
      --config "$TMP/p5c_${tag}.yaml" \
      --out "results/p5c_${tag}_voxsel.json"
}
for seed in 1234 2345 3456; do
  score "deform_s${seed}"; score "tcn192_s${seed}"
done
score tcn256_s1234

echo "=== PRIMARY: tcn192 vs deform (param-matched, paired) ==="
"$PY" scripts/compare_runs.py \
    --a results/p5c_deform_s1234_voxsel.json results/p5c_deform_s2345_voxsel.json results/p5c_deform_s3456_voxsel.json \
    --b results/p5c_tcn192_s1234_voxsel.json results/p5c_tcn192_s2345_voxsel.json results/p5c_tcn192_s3456_voxsel.json \
    --name-a "deform" --name-b "tcn192" --metric macro_f1

echo "=== MECHANISM: fragmentation / boundary F1 (seed 1234, both arms) ==="
for tag in deform_s1234 tcn192_s1234; do
  echo "--- $tag ---"
  PYTHONPATH=. "$PY" scripts/segment_diagnosis.py --manifest "$SEL" \
      --checkpoint "logs/p5c_${tag}/step${STEPS}.pt" --config "$TMP/p5c_${tag}.yaml" \
      --teacher-dir "/media/edabk/500GB Hard Disk/data_diar/voxconverse/diarizen_on_test" \
      | grep -aA8 "===== zipcount"
done
