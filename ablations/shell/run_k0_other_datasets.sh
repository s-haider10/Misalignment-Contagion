#!/bin/bash
# Run k0_baseline on the 4 datasets other than synthetic, n=50 each.
# Waits for the shadow_summary job (PID passed as $1) to finish first.
set -euo pipefail
cd /home/haider/Misalignment-Contagion
export PYTHONPATH=.
PY=/home/haider/Misalignment-Contagion/.venv/bin/python

WAIT_PID="${1:-}"
if [[ -n "$WAIT_PID" ]]; then
    echo "==== $(date) :: waiting for PID $WAIT_PID to finish ===="
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "==== $(date) :: PID $WAIT_PID finished, starting k0 runs ===="
fi

for ds in moral_stories harmbench_standard harmbench_contextual harmbench_copyright; do
    echo "==== $(date) :: k0 starting $ds ===="
    $PY -m misalignment_contagion.run_extra \
        --phase k0_baseline \
        --dataset "$ds" \
        --max-scenarios 50 \
        --single-server \
        --concurrency 8
    echo "==== $(date) :: k0 finished $ds ===="
done

echo "==== K0 ALL DATASETS DONE $(date) ===="
