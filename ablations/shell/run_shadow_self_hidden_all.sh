#!/bin/bash
# Run shadow_self_hidden_ablation across all 5 datasets, n=50 each.
# Waits for an upstream PID to finish first (passed as $1).
set -euo pipefail
cd /home/haider/Projects/active/misalignment-contagion-behavioral
export PYTHONPATH=.
PY=/home/haider/Projects/active/misalignment-contagion-behavioral/.venv/bin/python

WAIT_PID="${1:-}"
if [[ -n "$WAIT_PID" ]]; then
    echo "==== $(date) :: waiting for PID $WAIT_PID before starting self-hidden ablation ===="
    while [ -e "/proc/$WAIT_PID" ]; do sleep 60; done
    echo "==== $(date) :: PID $WAIT_PID finished, starting self-hidden runs ===="
fi

for ds in synthetic moral_stories harmbench_standard harmbench_contextual harmbench_copyright; do
    echo "==== $(date) :: self_hidden starting $ds ===="
    $PY -m misalignment_contagion.run_extra \
        --phase shadow_self_hidden_ablation \
        --dataset "$ds" \
        --max-scenarios 50 \
        --single-server \
        --concurrency 8
    echo "==== $(date) :: self_hidden finished $ds ===="
done

echo "==== SELF_HIDDEN ALL DONE $(date) ===="
