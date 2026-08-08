#!/bin/bash
# Wait for shadow_no_stance_ablation to finish, then start a graph_run
# worker on GPU 0 (uses existing vLLM on port 8000 — same server already
# running for the ablations).
set -euo pipefail
cd /home/haider/Misalignment-Contagion
export PYTHONPATH=.
PY=/home/haider/Misalignment-Contagion/.venv/bin/python

WAIT_PID="${1:-}"
LOG="logs/graph_run_gpu0.log"

if [[ -n "$WAIT_PID" ]]; then
    echo "[$(date)] waiting for PID $WAIT_PID before starting graph_run on GPU 0" >> "$LOG"
    while [ -e "/proc/$WAIT_PID" ]; do sleep 60; done
    echo "[$(date)] PID $WAIT_PID exited; starting graph_run on GPU 0" >> "$LOG"
fi

# vLLM on port 8000 (already running) — just attach a worker
"$PY" -m misalignment_contagion.graph_run \
    --dataset all \
    --n-scenarios 40 \
    --seed 42 \
    --model-key qwen-7b-instruct \
    --model-condition model_induced \
    --base-port 8000 \
    --concurrency 8 \
    --shard-tag gpu0 \
    --num-shards 1 \
    >> "$LOG" 2>&1
