#!/bin/bash
# Run primary_em (model_induced) for Llama-3.2-1B-Instruct on the synthetic
# dataset, matching the existing llama-8b-instruct primary_em scope:
#   50 scenarios x {0.1,0.2,0.3} x {chain,circle,star,fc}, seed 42 = 1050 trials.
#
# Waits for the vLLM server (port 8000) to be healthy before starting.
set -uo pipefail
cd /home/haider/Projects/active/misalignment-contagion-behavioral
export PYTHONPATH=.
PY=/home/haider/Projects/active/misalignment-contagion-behavioral/.venv/bin/python
PORT=8000
LOG_DIR=logs/llama1b_primary_em
mkdir -p "$LOG_DIR"

echo "[$(date)] waiting for vLLM health on :$PORT ..."
DEADLINE=$((SECONDS + 900))
until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
    if [[ $SECONDS -ge $DEADLINE ]]; then
        echo "[$(date)] FAIL: vLLM not healthy in 15 min"; tail -8 "$LOG_DIR/vllm_8000.log"; exit 1
    fi
    sleep 5
done
echo "[$(date)] vLLM healthy — starting primary_em run"

"$PY" -m misalignment_contagion.run \
    --phase primary_em \
    --dataset synthetic \
    --model-key llama-1b-instruct \
    --seeds 42 \
    --base-port "$PORT" \
    --n-servers 1 \
    --concurrency 24 \
    2>&1 | tee "$LOG_DIR/run.log"

echo "[$(date)] primary_em llama-1b-instruct synthetic DONE"
