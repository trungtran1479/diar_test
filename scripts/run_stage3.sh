#!/bin/bash
# Stage-3 full finetune run (see configs/zipcount_v2_stage3.yaml).
set -e
cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python

PYTHONPATH=. nohup "$PY" src/train.py \
    --config configs/zipcount_v2_stage3.yaml \
    --init-from logs/r6_v2_stage2b/best_macro_f1.pt \
    > logs/r7_v2_stage3_train.log 2>&1 &

echo "started pid $! -> tail -f logs/r7_v2_stage3_train.log"
