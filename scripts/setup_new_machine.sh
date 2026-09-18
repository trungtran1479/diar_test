#!/bin/bash
# =============================================================================
# ZipCount — setup on a NEW machine (downloads data + model, installs deps,
# rewrites all absolute paths baked into manifests/configs).
#
# Usage:
#   tar xzf zipcount_bundle_*.tar.gz && cd diar_new
#   bash scripts/setup_new_machine.sh            # full setup
#   DATA_ROOT=/mnt/big/zipcount_data bash scripts/setup_new_machine.sh
#
# Env overrides:
#   DATA_ROOT   where datasets are downloaded (default: <project>/data_ext)
#   VENV_DIR    virtualenv location            (default: <project>/venv)
#   SKIP_PIP=1  skip dependency installation
#   SKIP_DL=1   skip dataset downloads (only rewrite paths + smoke test)
#
# Disk needed under DATA_ROOT: ~75 GB (LibriSpeech 61GB + MUSAN 12GB + model)
# Verified source env: python 3.12 / torch 2.10.0+cu128 /
#   k2 1.24.4.dev20260306+cuda12.8.torch2.10.0 / lhotse 1.33.0
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data_ext}"
CONDA_ENV="${CONDA_ENV:-.zipformer}"
CONDA_PREFIX="${CONDA_PREFIX:-$HOME/miniconda3}"
PY="${PY:-$CONDA_PREFIX/envs/$CONDA_ENV/bin/python}"
ICEFALL_DIR="$PROJECT_ROOT/third_party/icefall"
MODEL_DIR="$DATA_ROOT/models/icefall-asr-librispeech-streaming-zipformer-2023-05-17/exp"

echo "== ZipCount setup =="
echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "DATA_ROOT    = $DATA_ROOT"
echo "PY           = $PY"
mkdir -p "$DATA_ROOT" "$MODEL_DIR"

if [ ! -x "$PY" ]; then
  echo "ERROR: Python not found: $PY"
  echo "Set PY=/path/to/python or CONDA_ENV=.zipformer before running this script."
  exit 1
fi

# # -----------------------------------------------------------------------------
# # 1. Python environment
# # -----------------------------------------------------------------------------
# if [ -z "${SKIP_PIP:-}" ]; then
#   echo "== [1/5] Python env =="
#   if [ ! -d "$VENV_DIR" ]; then
#     python3 -m venv "$VENV_DIR"
#   fi
#   PY="$VENV_DIR/bin/python"
#   "$PY" -m pip install -U pip

#   # Torch pinned to the version verified on the source machine.
#   # If this machine has a different CUDA, change the index-url accordingly
#   # (https://pytorch.org/get-started/locally/), and pick the matching k2 wheel
#   # from https://k2-fsa.github.io/k2/cuda.html
#   "$PY" -m pip install torch==2.10.0 torchaudio==2.10.0 \
#       --index-url https://download.pytorch.org/whl/cu128
#   "$PY" -m pip install "k2==1.24.4.dev20260306+cuda12.8.torch2.10.0" \
#       -f https://k2-fsa.github.io/k2/cuda.html \
#       || echo "WARNING: exact k2 wheel unavailable -> pick one matching your torch/CUDA from https://k2-fsa.github.io/k2/cuda.html"
#   "$PY" -m pip install -r "$PROJECT_ROOT/requirements.txt"
#   # icefall package (bundled copy — exact code the checkpoints were verified with)
#   "$PY" -m pip install -e "$ICEFALL_DIR" --no-deps
# else
#   PY="$VENV_DIR/bin/python"
# fi

# -----------------------------------------------------------------------------
# 2. Datasets (resumable: wget -c; re-running the script is safe)
# -----------------------------------------------------------------------------
if [ -z "${SKIP_DL:-}" ]; then
  echo "== [2/5] Datasets -> $DATA_ROOT =="
  cd "$DATA_ROOT"

  # --- LibriSpeech (sources for on-the-fly mixing): 6.3G + 23G + 0.3G ---
  for part in train-clean-100 train-clean-360 dev-clean; do
    if [ ! -d "$DATA_ROOT/LibriSpeech/$part" ]; then
      echo "-- LibriSpeech/$part"
      wget -c "https://www.openslr.org/resources/12/${part}.tar.gz"
      tar xzf "${part}.tar.gz" && rm -f "${part}.tar.gz"
    fi
  done

  # --- MUSAN (class-0 noise): 11G ---
  if [ ! -d "$DATA_ROOT/musan" ]; then
    echo "-- MUSAN"
    wget -c "https://www.openslr.org/resources/17/musan.tar.gz"
    tar xzf musan.tar.gz && rm -f musan.tar.gz
  fi

  # --- Backbone: k2-fsa LibriSpeech STREAMING Zipformer2 (causal), 265MB ---
  if [ ! -f "$MODEL_DIR/pretrained.pt" ]; then
    echo "-- streaming Zipformer2 checkpoint"
    wget -c -O "$MODEL_DIR/pretrained.pt" \
      "https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17/resolve/main/exp/pretrained.pt"
  fi

  # --- Optional eval sets (uncomment if needed) ---
  # mkdir -p "$DATA_ROOT/eval_sets/voxconverse" "$DATA_ROOT/eval_sets/libricount"
  # wget -c -O "$DATA_ROOT/eval_sets/voxconverse/dev_wav.zip"  https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip
  # wget -c -O "$DATA_ROOT/eval_sets/voxconverse/test_wav.zip" https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip
  # wget -c -O "$DATA_ROOT/eval_sets/voxconverse/rttms.zip"    https://github.com/joonson/voxconverse/archive/refs/heads/master.zip
  # wget -c -O "$DATA_ROOT/eval_sets/libricount/LibriCount10-0dB.zip" "https://zenodo.org/records/1216072/files/LibriCount10-0dB.zip?download=1"
  # AMI (~11GB, cho stage-2 finetune): https://groups.inf.ed.ac.uk/ami/download/
  # NOTSOFAR-1: https://github.com/microsoft/NOTSOFAR1-Challenge (Azure blob)
fi

# -----------------------------------------------------------------------------
# 3. Rewrite absolute paths (manifests + configs) — idempotent
# -----------------------------------------------------------------------------
echo "== [3/5] Rewrite paths =="
MARKER="$PROJECT_ROOT/.paths_rewritten"
if [ -f "$MARKER" ]; then
  echo "already rewritten ($MARKER exists), skipping"
else
  OLD_LS="/home/pc/v2t/data/en/mtasr_prep/data/librispeech/LibriSpeech"
  OLD_MUSAN="/home/pc/v2t/Add_noise_and_rir_to_speech/musan"
  OLD_LABELS="/home/pc/diar_new/data/librimix"
  OLD_ICEFALL="/home/pc/icefall"
  OLD_CKPT="/home/pc/v2t/models/icefall-asr-librispeech-streaming-zipformer-2023-05-17/exp/pretrained.pt"

  for f in "$PROJECT_ROOT"/data/librimix/train_manifest.json \
           "$PROJECT_ROOT"/data/librimix/val_manifest.json; do
    [ -f "$f" ] || { echo "MISSING $f"; exit 1; }
    sed -i "s|$OLD_LS|$DATA_ROOT/LibriSpeech|g; s|$OLD_MUSAN|$DATA_ROOT/musan|g; s|$OLD_LABELS|$PROJECT_ROOT/data/librimix|g" "$f"
    echo "rewrote $(basename "$f")"
  done

  for f in "$PROJECT_ROOT"/configs/zipcount_v1.yaml "$PROJECT_ROOT"/configs/zipcount_v2.yaml; do
    sed -i "s|$OLD_ICEFALL|$ICEFALL_DIR|g; s|$OLD_CKPT|$MODEL_DIR/pretrained.pt|g; s|/home/pc/diar_new|$PROJECT_ROOT|g" "$f"
    echo "rewrote $(basename "$f")"
  done
  touch "$MARKER"
fi

# -----------------------------------------------------------------------------
# 4. Sanity: every path type resolves
# -----------------------------------------------------------------------------
echo "== [4/5] Path sanity =="
"$PY" - <<EOF
import json, os, sys, random
random.seed(0)
bad = 0
for split in ["train", "val"]:
    lines = open("$PROJECT_ROOT/data/librimix/%s_manifest.json" % split).read().splitlines()
    for line in random.sample(lines, min(200, len(lines))):
        it = json.loads(line)
        for t in it.get("tracks", []):
            if not os.path.exists(t["source"]): print("MISSING", t["source"]); bad += 1
        if not os.path.exists(it["label_filepath"]): print("MISSING", it["label_filepath"]); bad += 1
print("sampled path check:", "OK" if bad == 0 else f"{bad} MISSING")
sys.exit(1 if bad else 0)
EOF

# -----------------------------------------------------------------------------
# 5. Smoke test: build real backbone + one forward pass on one real item
# -----------------------------------------------------------------------------
echo "== [5/5] Smoke test =="
cd "$PROJECT_ROOT"
PYTHONPATH=. "$PY" - <<EOF
import warnings, yaml, torch; warnings.filterwarnings("ignore")
from src.models.zipcount_v1 import build_model
from src.data.dataset import SpeakerCountDataset, collate_fn
with open("configs/zipcount_v1.yaml") as f: cfg = yaml.safe_load(f)
model = build_model(cfg)
ds = SpeakerCountDataset(cfg["data"]["train_manifest"])
batch = collate_fn([ds[0]])
with torch.no_grad():
    logits, h_lens = model(batch["features"], batch["feature_lens"])
print("SMOKE OK:", tuple(logits[0].shape), "device=cpu")
EOF

echo ""
echo "=============================================="
echo "Setup DONE. Train:"
echo "  conda activate $CONDA_ENV && cd $PROJECT_ROOT"
echo "  PYTHONPATH=. python src/train.py --config configs/zipcount_v1.yaml   # variant C"
echo "  PYTHONPATH=. python src/train.py --config configs/zipcount_v2.yaml   # variant E (v2)"
echo "=============================================="
