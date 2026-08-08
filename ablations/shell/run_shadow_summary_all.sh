#!/bin/bash
# Run shadow_summary_ablation across all 5 datasets, n=50 each.
set -euo pipefail
cd /home/haider/Misalignment-Contagion
export PYTHONPATH=.
PY=/home/haider/Misalignment-Contagion/.venv/bin/python

for ds in synthetic moral_stories harmbench_standard harmbench_contextual harmbench_copyright; do
    echo "==== $(date) :: starting $ds ===="
    $PY -m misalignment_contagion.run_extra \
        --phase shadow_summary_ablation \
        --dataset "$ds" \
        --max-scenarios 50 \
        --single-server \
        --concurrency 8
    echo "==== $(date) :: finished $ds ===="
done

echo "==== ALL DONE $(date) ===="
