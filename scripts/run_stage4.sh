#!/bin/bash
# Stage-4 run (r8): DiariZen's 6 real corpora. See configs/zipcount_v2_stage4.yaml.
set -e
cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python

PYTHONPATH=. nohup "$PY" src/train.py \
    --config configs/zipcount_v2_stage4.yaml \
    --init-from logs/r7_v2_stage3/best_macro_f1.pt \
    > logs/r8_v2_stage4_train.log 2>&1 &

echo "started pid $! -> tail -f logs/r8_v2_stage4_train.log"
