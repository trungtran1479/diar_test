#!/bin/bash
# Weight-space interpolation between v3 (alpha=0) and the step-400 fine-tune
# (alpha=1), swept per seed, to look for a better cross-domain operating point
# WITHOUT retraining.
#
# Motivation from the full benchmark: fine-tuning did not improve the model, it
# MOVED it. Confirmed real in both directions — VoxConverse +0.020 / vox_lock
# +0.022 / MSDWild +0.027, against AMI-test -0.017 / DipCo-dev -0.018. If the
# interpolation path bows OUTSIDE the chord joining the two endpoints, some
# alpha dominates both, and that is a result on its own.
#
# ---- PRE-REGISTERED BEFORE ANY ALPHA WAS SCORED ----
#
# selection set: a COMPOSITE DEV of three domains, each already used for
#   development, so none of it is a fresh holdout being burned:
#       AMI-SDM dev     meeting far-field, in-domain
#       DipCo-MDM dev   meeting far-field, never trained on
#       vox_sel         conversational in-the-wild
#   PRIMARY composite  = unweighted mean of the three macro-F1 scores.
#     Unweighted by DOMAIN, not by frames: the claim is about cross-domain
#     behaviour, so a long corpus must not outvote a short one.
#   SECONDARY          = min over the three (worst-domain), which is the
#     quantity a Pareto argument actually cares about.
#   alpha is chosen on the PRIMARY. The secondary is reported, not optimised.
#
# confirmatory: DipCo-MDM eval — never used for any decision to date. That is
#   the only genuinely untouched set left and it is read ONCE, after alpha is
#   fixed.
#
# NOT reused for selection: vox_lock and msdwild_manyval_LOCKED. They were
#   already spent measuring the frozen config. They may be reported afterwards,
#   but they have now been observed, and that must be disclosed rather than
#   quietly treated as fresh holdout.
#
# AMI-SDM test is likewise reporting-only: it influenced the early-stopping
#   decision earlier in this project, so it is no longer a clean holdout.
set -u
cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python
CFG=configs/zipcount_v4_distill.yaml
BASE=logs/r8_v3_diverse/best_macro_f1.pt
OUT=results/interp
TMPCK=/tmp/claude-1000/-home-edabk-hoangbpm/c22de1cf-7989-4e68-989d-f720938175c6/scratchpad/interp.pt
DATA="/media/edabk/500GB Hard Disk/data_diar"
mkdir -p "$OUT"

ALPHAS="0.25 0.50 0.75"     # 0.0 and 1.0 are the measured endpoints
SEEDS="1234 2345 3456"

for seed in $SEEDS; do
  B="logs/p2_ctrl_s${seed}/step400.pt"
  [ -f "$B" ] || { echo "MISSING $B" >&2; exit 1; }
  for a in $ALPHAS; do
    tag="a${a}_s${seed}"
    [ -s "$OUT/${tag}__voxsel.json" ] && { echo "skip $tag"; continue; }
    echo "=== interpolating $tag ==="
    "$PY" scripts/interp_ckpt.py --a "$BASE" --b "$B" --alpha "$a" --out "$TMPCK" || exit 1

    # composite dev, all three domains
    PYTHONPATH=. "$PY" src/eval_bench.py --bench ami --split dev \
        --config "$CFG" --checkpoint "$TMPCK" > "$OUT/${tag}__ami_dev.log" 2>&1
    PYTHONPATH=. "$PY" src/eval_bench.py --bench ami --split dev \
        --prefix dipco-mdm --manifest-dir data/manifests/dipco_mdm \
        --config "$CFG" --checkpoint "$TMPCK" > "$OUT/${tag}__dipco_dev.log" 2>&1
    PYTHONPATH=. "$PY" scripts/eval_per_recording.py \
        --manifest "$DATA/voxconverse/vox_sel.json" --checkpoint "$TMPCK" \
        --config "$CFG" --out "$OUT/${tag}__voxsel.json" > /dev/null 2>&1

    grep -ao "macroF1=[0-9.]*" "$OUT/${tag}__ami_dev.log"   | head -1 | sed "s/^/  ami_dev   /"
    grep -ao "macroF1=[0-9.]*" "$OUT/${tag}__dipco_dev.log" | head -1 | sed "s/^/  dipco_dev /"
    "$PY" -c "import json;print('  voxsel     macroF1=%.4f'%json.load(open('$OUT/${tag}__voxsel.json'))['pooled']['macro_f1'])"
    rm -f "$TMPCK"
  done
done

echo ""
echo "=== INTERPOLATION SWEEP DONE ==="
"$PY" scripts/summarise_interp.py --dir "$OUT" --bench-dir results/bench
