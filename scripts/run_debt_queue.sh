#!/bin/bash
# Serial GPU queue, revision 2 — the phase-8 chain is intentionally NOT here:
# after the methodology review it must be relaunched manually once its fixes
# are re-reviewed:
#   PYTHONPATH=. python scripts/phase8_chain_driver.py 2>&1 | tee logs/phase8_chain.log
# Stages are independent; failures are collected and the queue exits nonzero
# if anything failed (revision 1 always printed FINISHED and exited 0).
cd "$(dirname "$0")/.."
mkdir -p logs
FAILED=""

echo "[queue] $(date -Is) preflight: full test suite"
if ! PYTHONPATH=. /home/edabk/miniconda3/envs/.zipformer/bin/python -m pytest tests/ -q \
     > logs/debt_queue_tests.log 2>&1; then
  echo "[queue] FATAL: test suite failed — refusing to start (see logs/debt_queue_tests.log)"
  exit 1
fi
tail -1 logs/debt_queue_tests.log

for stage in run_context_oracle run_stack_probe; do
  echo "[queue] $(date -Is) START $stage"
  if bash "scripts/${stage}.sh" > "logs/${stage}_queue.log" 2>&1; then
    echo "[queue] $(date -Is) DONE  $stage"
  else
    echo "[queue] $(date -Is) FAILED $stage (see logs/${stage}_queue.log)"
    FAILED="$FAILED $stage"
  fi
done

if [ -n "$FAILED" ]; then
  echo "[queue] $(date -Is) FINISHED WITH FAILURES:$FAILED"
  exit 1
fi
echo "[queue] $(date -Is) ALL STAGES OK (phase-8 chain held for re-review)"
