#!/bin/bash
# Multi-corpus paper benchmark of the two systems the Phase-8 investigation
# actually settled on (head-only / frozen backbone, 3 seeds each):
#
#   mask  = TCN head + stack_input_mask [1,1,1,0,0,0]  (main system: adopted
#           via R3 in the preregistered graduation, +0.0043 vs root, 3/3
#           seeds positive; see STACK_MASK_GRADUATION_PREREG.md)
#   root  = TCN head, all six stacks, unmasked          (ablation row: the
#           immutable Phase-8 chain root, "w/o stack selection")
#
# The pyramid+mask candidate is NOT here: the rev-8 chain's final gate did
# not pass (+0.0006 vs root, need +0.010), so the TCN family remains the
# promoted system for the paper.
#
# Scope (stated explicitly, not silently narrowed): boundary AP / recall@FP
# is computed on vox_sel ONLY (the set every chain/graduation gate already
# used) -- extending it to the full-size corpora is unbounded extra compute
# for a metric this project has already exercised extensively. Segmental
# edit score / fragmentation / calibration run on all four window-JSON
# corpora, where window contiguity was verified (ids are ..._w0000, _w0001,
# ... in fixed 30s hops, confirmed against the manifests before writing
# eval_sequence_metrics.py). AMI/DipCo/LibriCount use the existing
# src/eval_bench.py harness (frame metrics only -- its windowing was not
# re-verified for sequence-metric contiguity).
set -u
cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python
OUT=results/bench_paper
mkdir -p "$OUT"
DATA="/media/edabk/500GB Hard Disk/data_diar"

SEEDS="1234 2345 3456"

ckpt_for() { # system seed
  if [ "$1" = "mask" ]; then echo "logs/stackgrad_pre012_s$2/step3000.pt"
  else echo "logs/p8a_average_s$2/step3000.pt"; fi
}
cfg_for() { # system seed
  if [ "$1" = "mask" ]; then echo "artifacts/stack_mask_graduation/stackgrad_pre012_s$2.yaml"
  else echo "artifacts/phase8a_alignment/p8a_average_s$2.yaml"; fi
}

for system in mask root; do
  for seed in $SEEDS; do
    name="${system}_s${seed}"
    ck="$(ckpt_for "$system" "$seed")"
    cfg="$(cfg_for "$system" "$seed")"
    [ -f "$ck" ] || { echo "MISSING checkpoint $ck" >&2; continue; }
    [ -f "$cfg" ] || { echo "MISSING config $cfg" >&2; continue; }
    echo "############ $name ############"

    # --- AMI / DipCo / LibriCount (existing lhotse-based harness) ---
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
          --config "$cfg" --checkpoint "$ck" > "$f" 2>&1 || echo "  FAILED (see $f)"
    done
    f="$OUT/${name}__libricount.log"
    if [ ! -s "$f" ]; then
      echo ">>> $name / libricount"
      PYTHONPATH=. "$PY" src/eval_bench.py --bench libricount \
          --config "$cfg" --checkpoint "$ck" > "$f" 2>&1 || echo "  FAILED (see $f)"
    fi

    # --- window-JSON corpora: per-recording F1/MAE + sequence metrics ---
    for spec in "voxsel:$DATA/voxconverse/vox_sel.json" \
                "voxfull:$DATA/voxconverse/voxconverse_test_eval.json" \
                "voxlock:$DATA/voxconverse/vox_lock.json" \
                "msdwild:$DATA/msdwild/msdwild_manyval_LOCKED.json"; do
      IFS=: read -r tag mf <<< "$spec"
      [ -f "$mf" ] || { echo "  no manifest $mf"; continue; }

      j="$OUT/${name}__${tag}_perrec.json"
      # vox_sel per-recording F1/MAE already exists from the graduation /
      # phase8a runs (identical manifest+checkpoint+config) -- reuse instead
      # of recomputing.
      existing=""
      if [ "$tag" = "voxsel" ]; then
        if [ "$system" = "mask" ]; then existing="results/stackgrad_pre012_s${seed}_voxsel.json"
        else existing="results/p8a_average_s${seed}_voxsel.json"; fi
      fi
      if [ -n "$existing" ] && [ -s "$existing" ] && [ ! -s "$j" ]; then
        cp "$existing" "$j"
        echo "reuse $existing -> $j"
      elif [ ! -s "$j" ]; then
        echo ">>> $name / $tag per-recording"
        PYTHONPATH=. "$PY" scripts/eval_per_recording.py \
            --manifest "$mf" --checkpoint "$ck" --config "$cfg" \
            --out "$j" > "$OUT/${name}__${tag}_perrec.log" 2>&1 || echo "  FAILED"
      else
        echo "skip $j"
      fi

      s="$OUT/${name}__${tag}_seq.json"
      if [ ! -s "$s" ]; then
        echo ">>> $name / $tag sequence metrics"
        PYTHONPATH=. "$PY" scripts/eval_sequence_metrics.py \
            --manifest "$mf" --checkpoint "$ck" --config "$cfg" \
            --out "$s" > "$OUT/${name}__${tag}_seq.log" 2>&1 || echo "  FAILED"
      else
        echo "skip $s"
      fi
    done

    # --- boundary AP / recall@FP-budget, vox_sel only (see header) ---
    b="$OUT/${name}__voxsel_boundary.json"
    if [ ! -s "$b" ]; then
      echo ">>> $name / vox_sel boundary AP"
      PYTHONPATH=. "$PY" scripts/eval_boundary_ap.py \
          --manifest "$DATA/voxconverse/vox_sel.json" --checkpoint "$ck" \
          --config "$cfg" --out "$b" > "$OUT/${name}__voxsel_boundary.log" 2>&1 \
          || echo "  FAILED"
    else
      echo "skip $b"
    fi
  done
done

echo ""
echo "=== ALL PAPER BENCHMARK RUNS DONE ==="
