#!/bin/bash
# Sequentially runs the remaining loss-ablation training jobs, cleaning up
# intermediate checkpoints after each so disk never holds more than one
# run's checkpoints at a time. Emits ALL_DONE on success, or an error
# marker and stops on the first failure.
set -u
cd /home/edabk/hoangbpm/diar/diar_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate .zipformer

CONFIGS=(
  lossabl_countonly_s1234
  lossabl_countonly_s2345
  lossabl_countonly_s3456
  lossabl_auxvad_s1234
  lossabl_auxvad_s2345
  lossabl_auxvad_s3456
  lossabl_ordinal_s1234
  lossabl_ordinal_s2345
  lossabl_ordinal_s3456
)

for name in "${CONFIGS[@]}"; do
  if [ -f "logs/${name}/step3000.pt" ]; then
    echo "############ $name (already done, skipping) ############"
    continue
  fi
  echo "############ $name ############"
  PYTHONPATH=. python src/train.py \
      --config artifacts/loss_ablation/${name}.yaml \
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
