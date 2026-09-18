#!/bin/bash
# Full benchmark of the SELECTED configuration, over all three seeds.
#
# Selected config (fixed before this ran, by Phase 1-3): uniform LR 3e-5,
# lambda_kd=0, endpoint step 400. KD is OFF because two mask designs, six paired
# runs each, both failed to beat the control — `agree` was null, `agree_uncertain`
# was negative with a dose-response. Nothing here is being tuned; this measures
# the model we already committed to.
#
# All three seeds are reported, never the best one. Within-arm seed spread on
# VoxConverse measured ~0.02 macro-F1, so a single-seed benchmark number would
# be indistinguishable from cherry-picking.
#
# Corpora, by how much the model has seen them:
#   AMI-SDM dev/test    AMI train was in the training mix (in-domain)
#   VoxConverse test    VoxConverse train was in the mix (in-domain)
#   MSDWild many.val    LOCKED, 0 recordings overlap the 2,438 training ones
#   vox_lock            LOCKED, held out from every previous decision
#   DipCo-MDM dev/eval  NEVER trained on, never evaluated on — true out-of-domain
#   LibriCount          synthetic, clip-level, out of domain
#
# The locked sets are spent HERE, once, on a configuration that is already
# frozen. That is what they were reserved for.
set -u
cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python
CFG=configs/zipcount_v4_distill.yaml
OUT=results/bench
mkdir -p "$OUT"

DATA="/media/edabk/500GB Hard Disk/data_diar"

declare -A MODELS=(
  [v3_base]=logs/r8_v3_diverse/best_macro_f1.pt
  [ft_s1234]=logs/p2_ctrl_s1234/step400.pt
  [ft_s2345]=logs/p2_ctrl_s2345/step400.pt
  [ft_s3456]=logs/p2_ctrl_s3456/step400.pt
)

for name in v3_base ft_s1234 ft_s2345 ft_s3456; do
  ck="${MODELS[$name]}"
  [ -f "$ck" ] || { echo "MISSING $ck" >&2; continue; }

  # --- lhotse-manifest benchmarks (long recordings, windowed internally) ---
  for spec in "ami:ami-sdm:data/manifests/ami:dev" \
              "ami:ami-sdm:data/manifests/ami:test" \
              "dipco:dipco-mdm:data/manifests/dipco_mdm:dev" \
              "dipco:dipco-mdm:data/manifests/dipco_mdm:eval"; do
    IFS=: read -r bench prefix mdir split <<< "$spec"
    f="$OUT/${name}__${prefix}_${split}.log"
    [ -s "$f" ] && { echo "skip $f"; continue; }
    echo ">>> $name / $prefix $split"
    PYTHONPATH=. "$PY" src/eval_bench.py --bench ami --split "$split" \
        --prefix "$prefix" --manifest-dir "$mdir" \
        --config "$CFG" --checkpoint "$ck" > "$f" 2>&1 || echo "  FAILED (see $f)"
  done

  f="$OUT/${name}__libricount.log"
  if [ ! -s "$f" ]; then
    echo ">>> $name / libricount"
    PYTHONPATH=. "$PY" src/eval_bench.py --bench libricount \
        --config "$CFG" --checkpoint "$ck" > "$f" 2>&1 || echo "  FAILED (see $f)"
  fi

  # --- window-manifest benchmarks (per-recording, for bootstrap later) ---
  for spec in "voxfull:$DATA/voxconverse/voxconverse_test_eval.json" \
              "voxlock:$DATA/voxconverse/vox_lock.json" \
              "msdwild:$DATA/msdwild/msdwild_manyval_LOCKED.json"; do
    IFS=: read -r tag mf <<< "$spec"
    j="$OUT/${name}__${tag}.json"
    [ -s "$j" ] && { echo "skip $j"; continue; }
    [ -f "$mf" ] || { echo "  no manifest $mf"; continue; }
    echo ">>> $name / $tag"
    PYTHONPATH=. "$PY" scripts/eval_per_recording.py \
        --manifest "$mf" --checkpoint "$ck" --config "$CFG" \
        --out "$j" > "$OUT/${name}__${tag}.log" 2>&1 || echo "  FAILED"
  done
done

echo ""
echo "=== ALL BENCHMARKS DONE ==="
"$PY" scripts/summarise_benchmark.py --dir "$OUT"
