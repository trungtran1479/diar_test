#!/bin/bash
# v4: distillation from DiariZen (offline teacher) into the causal streaming
# student, on top of v3's diverse-domain data. See configs/zipcount_v4_distill.yaml.
set -e
cd "$(dirname "$0")/.."

PY=/home/edabk/miniconda3/envs/.zipformer/bin/python
CONFIG=configs/zipcount_v4_distill.yaml
INIT=logs/r8_v3_diverse/best_macro_f1.pt

# Fail loudly rather than silently training without KD.
grep -q '^distill:' "$CONFIG" || { echo "ERROR: $CONFIG has no distill: block"; exit 1; }
TEACHER_DIR=$("$PY" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['distill']['teacher_dir'])")
[ -d "$TEACHER_DIR" ] || { echo "ERROR: teacher_dir not found: $TEACHER_DIR"; exit 1; }
# find, not a glob: the teacher dir holds ~55k files and `ls *.npy` blows the
# argument list, which made this guard report zero and refuse a valid run
N=$(find "$TEACHER_DIR" -maxdepth 1 -name '*.npy' | wc -l)
echo "teacher posteriors available: $N"
[ "$N" -gt 0 ] || { echo "ERROR: no teacher posteriors in $TEACHER_DIR"; exit 1; }

PYTHONPATH=. nohup "$PY" src/train.py \
    --config "$CONFIG" \
    --init-from "$INIT" \
    > logs/r9_v4_distill_train.log 2>&1 &

echo "started pid $! -> tail -f logs/r9_v4_distill_train.log"
