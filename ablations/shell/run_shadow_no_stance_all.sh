#!/bin/bash
# Run shadow_no_stance_ablation across all 5 datasets, n=50 each.
# Waits for an upstream PID to finish first (passed as $1).
set -euo pipefail
cd /home/haider/Misalignment-Contagion
export PYTHONPATH=.
PY=/home/haider/Misalignment-Contagion/.venv/bin/python

WAIT_PID="${1:-}"
if [[ -n "$WAIT_PID" ]]; then
    echo "==== $(date) :: waiting for PID $WAIT_PID before starting no-stance ablation ===="
    while [ -e "/proc/$WAIT_PID" ]; do sleep 60; done
    echo "==== $(date) :: PID $WAIT_PID finished, starting no-stance runs ===="
fi

for ds in synthetic moral_stories harmbench_standard harmbench_contextual harmbench_copyright; do
    echo "==== $(date) :: no_stance starting $ds ===="
    $PY -m misalignment_contagion.run_extra \
        --phase shadow_no_stance_ablation \
        --dataset "$ds" \
        --max-scenarios 50 \
        --single-server \
        --concurrency 8
    echo "==== $(date) :: no_stance finished $ds ===="
done

echo "==== NO_STANCE ALL DONE $(date) ===="
