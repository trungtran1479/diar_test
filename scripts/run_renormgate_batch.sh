#!/bin/bash
# Sequentially runs the 3 renormalized-gate seeds, cleaning up intermediate
# checkpoints after each. Emits ALL_DONE on success.
set -u
cd /home/edabk/hoangbpm/diar/diar_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate .zipformer

CONFIGS=(
  renormgate_s1234
  renormgate_s2345
  renormgate_s3456
)

for name in "${CONFIGS[@]}"; do
  if [ -f "logs/${name}/step3000.pt" ]; then
    echo "############ $name (already done, skipping) ############"
    continue
  fi
  echo "############ $name ############"
  PYTHONPATH=. python src/train.py \
      --config artifacts/renorm_gate/${name}.yaml \
      > logs/${name}_run.log 2>&1
  rc=$?
  echo "$name exit=$rc"
  ls logs/${name}/step*.pt 2>/dev/null | grep -v step3000.pt | xargs -r rm -f
  if [ $rc -ne 0 ]; then
    echo "BATCH_FAILED at $name"
    exit 1
  fi
done
echo "ALL_DONE"
