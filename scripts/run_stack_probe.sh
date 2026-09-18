#!/bin/bash
# STACK PROBE (doc §10 phase A, runs A0-A5) — which Zipformer stack actually
# carries the speaker-count information? The v1->v2 jump added the multiscale
# hypercolumn without ever measuring this, and a reviewer will ask.
#
# Pre-registered before running:
#   arms      A0..A5 = TCN-192 head reading ONE stack via head.stack_indices.
#             The encoder path is byte-identical across arms (the head slices
#             its own input), so the ONLY factor is which stack the head sees.
#   control   A6 = all six stacks, static gate = the existing p8a_average_s1234
#             run — unlike old p5c it shares this script's RNG-reset and AMP
#             scale 4096, so the reuse is protocol-clean.
#   protocol  frozen v3 backbone, init_backbone_only, 3000 steps @1e-3 cosine,
#             batch 24, seed 1234 ONLY, endpoint LOCKED step 3000.
#   readout   vox_sel pooled macro-F1 + per-class F1 + OSD per arm. This is a
#             DIAGNOSTIC ONLY, never an architecture claim: single seed, and
#             the arms are NOT capacity-matched (head params range ~0.94M to
#             ~1.0M with stack width, vs 1.285M for all-6) — so a result mixes
#             stack information with input dim/capacity/initialization. Only
#             differences >> 0.02 and per-class PATTERNS may be read, and any
#             candidate finding must graduate to a matched, multi-seed test.
#   stacks    dims [192,256,384,512,384,256], downsampling [1,2,4,8,4,2]
#             => stack 3 is the 12.5 Hz bottleneck, stacks 0/5 are 50 Hz.
set -e
cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python
INIT=logs/r8_v3_diverse/best_macro_f1.pt
STEPS=3000
ART=artifacts/stack_probe
SEL="/media/edabk/500GB Hard Disk/data_diar/voxconverse/vox_sel.json"
mkdir -p "$ART" results

need_gb=16
avail_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "$avail_gb" -ge "$need_gb" ] || { echo "FATAL: ${avail_gb}G free" >&2; exit 1; }
[ -s results/p8a_average_s1234_voxsel.json ] || {
  echo "FATAL: A6 control results/p8a_average_s1234_voxsel.json missing" >&2; exit 1; }

run_one () {
  local idx=$1
  local tag="probe_stack${idx}_s1234"
  if [ -s "results/${tag}_voxsel.json" ] && [ -s "$ART/${tag}.lineage.json" ]; then
    if "$PY" scripts/check_lineage_sidecar.py "$ART/${tag}.lineage.json" \
         "$ART/${tag}.yaml" "$INIT"; then
      echo "[skip] $tag (lineage verified)"; return
    fi
    echo "[rerun] $tag: lineage drift"
    mv "results/${tag}_voxsel.json" "results/${tag}_voxsel.json.stale.$(date +%s)"
  elif [ -s "results/${tag}_voxsel.json" ]; then
    echo "[rerun] $tag: result without sidecar"
    mv "results/${tag}_voxsel.json" "results/${tag}_voxsel.json.stale.$(date +%s)"
  fi
  local cfg="$ART/${tag}.yaml"
  "$PY" - "$cfg" "$idx" "$tag" "$STEPS" <<'PYEOF'
import sys, yaml
cfg_out, idx, tag, steps = sys.argv[1:5]
c = yaml.safe_load(open("configs/zipcount_v4_distill.yaml"))
c["distill"]["lambda_kd"] = 0.0
c["loss"]["lambda_boundary_smooth"] = 0.0
c["model"]["head"] = {"type": "tcn_ordinal", "d_model": 192, "dropout": 0.1,
                      "stack_indices": [int(idx)]}
t = c["training"]
t["max_steps"] = int(steps)
t["seed"] = 1234
t["freeze_backbone_all"] = True
t["init_backbone_only"] = True
t["reset_rng_after_init"] = True
t["amp_init_scale"] = 4096.0
t["max_amp_overflows"] = 0   # paired vs A6 control: overflow invalidates
t["lr"] = 1e-3
t.pop("backbone_lr", None); t.pop("head_lr", None)
t["eval_interval"] = 300
t["eval_interval_early"] = 300
t["early_phase_steps"] = int(steps)
t["save_every_eval"] = True
t["log_dir"] = f"logs/{tag}"
yaml.safe_dump(c, open(cfg_out, "w"), sort_keys=False)
PYEOF
  echo "=== $tag (stack $idx only) ==="
  rm -rf "logs/${tag}"
  PYTHONPATH=. "$PY" src/train.py --config "$cfg" --init-from "$INIT" \
      > "logs/${tag}.log" 2>&1
  find "logs/${tag}" -type f ! -name "step${STEPS}.pt" ! -name 'events*' -delete
  PYTHONPATH=. "$PY" scripts/eval_per_recording.py \
      --manifest "$SEL" --checkpoint "logs/${tag}/step${STEPS}.pt" \
      --config "$cfg" --out "results/${tag}_voxsel.json"
  "$PY" scripts/check_lineage_sidecar.py --write "$ART/${tag}.lineage.json" \
      "$cfg" "$INIT"
}

for idx in 0 1 2 3 4 5; do run_one "$idx"; done

echo ""
echo "=== STACK PROBE TABLE (vox_sel pooled, step3000, seed 1234) ==="
"$PY" - <<'PYEOF'
import json
DIMS = [192, 256, 384, 512, 384, 256]
DS = [1, 2, 4, 8, 4, 2]
rows = []
for i in range(6):
    d = json.load(open(f"results/probe_stack{i}_s1234_voxsel.json"))["pooled"]
    rows.append((f"A{i} stack{i} ({50//DS[i]}Hz,{DIMS[i]}d)", d))
d = json.load(open("results/p8a_average_s1234_voxsel.json"))["pooled"]
rows.append(("A6 all-6 static (=p8a_avg)", d))
print(f"{'arm':<28}{'macro':>8}{'f1_0':>8}{'f1_1':>8}{'f1_2':>8}{'f1_3':>8}{'osd':>8}")
for name, p in rows:
    print(f"{name:<28}{p['macro_f1']:8.4f}{p['f1_0']:8.4f}{p['f1_1']:8.4f}"
          f"{p['f1_2']:8.4f}{p['f1_3']:8.4f}{p['osd_f1']:8.4f}")
print("\nsingle seed: interpret only gaps >> 0.02, and per-class patterns.")
PYEOF
