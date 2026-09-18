#!/bin/bash
# CONTEXT ORACLE — the longest-owed experiment: how much of the 0.10-0.13 gap
# to the offline DiariZen teacher is the CAUSALITY CONSTRAINT, and where does
# future context matter — head or encoder?
#
# Pre-registered before running:
#   three configs (doc §10 phase D), all on the TCN system, all initialized
#   from logs/p5c_tcn192_s1234/step3000.pt and full-fine-tuned with the exact
#   Phase-6c graduation regime (uniform 3e-5, 800 steps, legacy loss
#   lambda_smooth=0.02, KD off, endpoint LOCKED step 400):
#     O0  causal encoder + causal head        (rerun, same protocol/revision)
#     O1  causal encoder + NON-causal head    (symmetric-pad TCN, +-2.5s view)
#     O2  NON-causal encoder + NON-causal head (full offline oracle)
#   O1-O0 = value of future context in the DECODER only.
#   O2-O1 = additional value of future context in the ENCODER.
#   O2-O0 = the full price of streaming for this architecture.
#
#   ONE seed (1234). This is an ORACLE, not a competition entry: the question
#   is whether the gap moves by ~0.05-0.10 (an order of magnitude above the
#   0.02 seed floor) or by millipoints. Either answer is decisive at n=1; a
#   millipoint answer would say causality is NOT the bottleneck and the
#   remaining gap is capacity/data. Seeds 2345/3456 are added only if the
#   result lands awkwardly between (0.02-0.05).
#
#   Non-causal checkpoints are DIAGNOSTIC ONLY: they must never be presented
#   as the streaming system, and forward_streaming on them raises by design.
set -e
cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python
INIT=logs/p5c_tcn192_s1234/step3000.pt
STEPS=800
ART=artifacts/context_oracle
SEL="/media/edabk/500GB Hard Disk/data_diar/voxconverse/vox_sel.json"
mkdir -p "$ART" results

need_gb=16
avail_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "$avail_gb" -ge "$need_gb" ] || { echo "FATAL: ${avail_gb}G free" >&2; exit 1; }
[ -f "$INIT" ] || { echo "FATAL: missing $INIT" >&2; exit 1; }

# preflight: chunk=4096 covers a window only if every window fits inside it
"$PY" - <<'PYEOF'
import json
LIMIT_S = 4096 / 50.0        # encoder runs at 50 Hz internally
for mf in ["/media/edabk/500GB Hard Disk/data_diar/manifests/train_manifest_v3.json",
           "/media/edabk/500GB Hard Disk/data_diar/voxconverse/vox_sel.json"]:
    worst = max(json.loads(l)["duration"] for l in open(mf))
    assert worst < LIMIT_S, f"{mf}: window {worst}s >= {LIMIT_S}s breaks the chunk-4096 oracle"
    print(f"preflight OK: {mf.split('/')[-1]} max window {worst:.1f}s < {LIMIT_S:.1f}s")
PYEOF
# O0 is RERUN under this exact script (second review: reusing historical p6c
# mixed causality with AMP regime 65536-vs-4096 and source revision).

run_one () {
  local arm=$1 head_causal=$2 enc_causal=$3 seed=$4
  local tag="oracle_${arm}_s${seed}"
  local init="logs/p5c_tcn192_s${seed}/step3000.pt"
  [ -f "$init" ] || { echo "FATAL: missing $init" >&2; exit 1; }
  # 5th review: existence alone must not gate reuse — the sidecar binds the
  # arm config AND the init checkpoint; any drift forces a rerun.
  if [ -s "results/${tag}_voxsel.json" ] && [ -s "$ART/${tag}.lineage.json" ]; then
    if "$PY" scripts/check_lineage_sidecar.py "$ART/${tag}.lineage.json" \
         "$ART/${tag}.yaml" "$init"; then
      echo "[skip] $tag (lineage verified)"; return
    fi
    echo "[rerun] $tag: lineage drift"
    mv "results/${tag}_voxsel.json" "results/${tag}_voxsel.json.stale.$(date +%s)"
  elif [ -s "results/${tag}_voxsel.json" ]; then
    echo "[rerun] $tag: result without sidecar"
    mv "results/${tag}_voxsel.json" "results/${tag}_voxsel.json.stale.$(date +%s)"
  fi
  local cfg="$ART/${tag}.yaml"
  "$PY" - "$cfg" "$head_causal" "$enc_causal" "$tag" "$STEPS" "$seed" <<'PYEOF'
import sys, yaml
cfg_out, head_causal, enc_causal, tag, steps, seed = sys.argv[1:7]
c = yaml.safe_load(open("configs/zipcount_v4_distill.yaml"))
c["distill"]["lambda_kd"] = 0.0
c["loss"]["lambda_boundary_smooth"] = 0.0
c["model"]["head"] = {"type": "tcn_ordinal", "d_model": 192, "dropout": 0.1,
                      "causal": head_causal == "true"}
if enc_causal == "false":
    # Full-context encoder WITHOUT changing module topology: causal=True keeps
    # ChunkCausalDepthwiseConv1d (setting causal=False swaps it for Conv1d and
    # silently random-inits 64 temporal convs: missing=128/unexpected=320 on
    # load — measured). chunk_size=-1 / left_context=-1 gives every frame the
    # whole recording as context through the SAME modules, so the checkpoint
    # loads missing=0/unexpected=0 and O2-O1 isolates future context alone.
    # chunk=-1 takes a DIFFERENT Python-RNG path in the recipe (one
    # random.choice instead of two), desynchronising dropout streams vs
    # O0/O1 and mixing context with stochastic regularisation (3rd review).
    # chunk=4096 keeps the exact two-draw code path while still covering any
    # whole window (preflight below asserts every window < 4096 encoder
    # frames = 81.9 s).
    c["model"]["encoder"]["chunk_size"] = "4096"
    c["model"]["encoder"]["left_context_frames"] = "-1"
t = c["training"]
t["max_steps"] = int(steps)
t["seed"] = int(seed)
# O1 died at step 193 to a single AMP overflow at the default init_scale
# 65536: the non-causal head shifts early losses upward. Use the P8A scale
# and rely on the (new) bounded overflow budget in train.py.
t["amp_init_scale"] = 4096.0
# 3rd review: even single-seed, the three arms are PAIRED via shared batch
# order; one scaler skip shifts that order. Any overflow now aborts the arm.
t["max_amp_overflows"] = 0
t["expected_init_missing"] = []           # strict load: any topology drift aborts
t["require_no_unexpected_init_keys"] = True
t["eval_interval"] = 200
t["eval_interval_early"] = 200
t["early_phase_steps"] = int(steps)
t["save_every_eval"] = True
t["log_dir"] = f"logs/{tag}"
yaml.safe_dump(c, open(cfg_out, "w"), sort_keys=False)
PYEOF
  echo "=== $tag (head_causal=$head_causal enc_causal=$enc_causal) ==="
  rm -rf "logs/${tag}"
  PYTHONPATH=. "$PY" src/train.py --config "$cfg" --init-from "$init" \
      > "logs/${tag}.log" 2>&1
  grep -aE "^\[val@400\]" "logs/${tag}.log" | sed 's/acc=[^ ]* mae=[^ ]* //'
  find "logs/${tag}" -type f ! -name 'step400.pt' ! -name 'events*' -delete
  PYTHONPATH=. "$PY" scripts/eval_per_recording.py \
      --manifest "$SEL" --checkpoint "logs/${tag}/step400.pt" \
      --config "$cfg" --out "results/${tag}_voxsel.json"
  "$PY" scripts/check_lineage_sidecar.py --write "$ART/${tag}.lineage.json" \
      "$cfg" "$init"
}

# Preregistration: the s1234 encoder contrast landed at +0.0241, inside the
# 0.02-0.05 add-seeds zone, so the FULL ladder runs at all three seeds. Each
# seed inits from its own p5c head-only checkpoint (the phase-6c convention);
# within a seed the three arms share init and batch order.
for seed in 1234 2345 3456; do
  run_one causal true  true  "$seed"   # O0
  run_one headnc false true  "$seed"   # O1
  run_one fullnc false false "$seed"   # O2
done

echo ""
echo "=== CONTEXT ORACLE LADDER (vox_sel pooled macro-F1, step400) ==="
"$PY" - <<'PYEOF'
import json, os
import numpy as np
def m(p):
    return json.load(open(p))["pooled"]["macro_f1"] if os.path.exists(p) else None
seeds = [1234, 2345, 3456]
rows = {a: [m(f"results/oracle_{a}_s{s}_voxsel.json") for s in seeds]
        for a in ("causal", "headnc", "fullnc")}
have = [i for i, s in enumerate(seeds) if all(rows[a][i] is not None for a in rows)]
print(f"{'config':<26}" + "".join(f"  s{seeds[i]:>5}" for i in have) + "    mean")
for a, label in [("causal", "O0 causal"), ("headnc", "O1 head-noncausal"),
                 ("fullnc", "O2 full offline")]:
    vals = [rows[a][i] for i in have]
    print(f"{label:<26}" + "".join(f"  {v:6.4f}" for v in vals)
          + f"  {np.mean(vals):6.4f}")
for name, hi, lo in [("head ctx (O1-O0)", "headnc", "causal"),
                     ("encoder ctx (O2-O1)", "fullnc", "headnc"),
                     ("streaming price (O2-O0)", "fullnc", "causal")]:
    d = [rows[hi][i] - rows[lo][i] for i in have]
    sign = "consistent" if all(x > 0 for x in d) or all(x < 0 for x in d) else "MIXED SIGN"
    print(f"{name:<26} per-seed {['%+.4f' % x for x in d]}  mean {np.mean(d):+.4f}  [{sign}]")
print("\nteacher reference on this set: DiariZen 0.6595 (offline)")
PYEOF
