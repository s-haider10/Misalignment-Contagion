#!/bin/bash
# k0_baseline on 4 datasets using Qwen-0.5B-Instruct on GPU 1 (port 8001).
set -euo pipefail
cd /home/haider/Projects/active/misalignment-contagion-behavioral
export PYTHONPATH=.
PY=/home/haider/Projects/active/misalignment-contagion-behavioral/.venv/bin/python

for ds in moral_stories harmbench_standard harmbench_contextual harmbench_copyright; do
    echo "==== $(date) :: k0/0.5B starting $ds ===="
    $PY -m misalignment_contagion.run_extra \
        --phase k0_baseline \
        --dataset "$ds" \
        --model-key qwen-0.5b-instruct \
        --max-scenarios 50 \
        --single-server \
        --base-port 8001 \
        --concurrency 8
    echo "==== $(date) :: k0/0.5B finished $ds ===="
done

echo "==== K0/0.5B ALL DONE $(date) ===="
