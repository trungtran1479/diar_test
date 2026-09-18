#!/bin/bash
# Stage-2b domain adaptation run (see configs/zipcount_v2_stage2b.yaml).
set -e
cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python

PYTHONPATH=. nohup "$PY" src/train.py \
    --config configs/zipcount_v2_stage2b.yaml \
    --init-from logs/r5_v2_stage2/best_macro_f1.pt \
    > logs/r6_v2_stage2b_train.log 2>&1 &

echo "started pid $! -> tail -f logs/r6_v2_stage2b_train.log"
