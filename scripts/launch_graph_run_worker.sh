#!/bin/bash
# Launch a vLLM server + graph_run worker on a specific GPU.
#
# Args:
#   $1 = GPU id (e.g., 0, 1, 2, 3)
#   $2 = port (default 8000 + gpu_id*10, e.g. 8000, 8010, 8020, 8030)
#   $3 = shard-tag (default "gpu${gpu_id}")
#
# vLLM serves Qwen-7B-Instruct base + ModelOrganismsForEM LoRA "misaligned"
# on the same port (single-server layout). Then a worker iterates the 5
# datasets and runs graph trials, claiming each one against shared
# claim files so multiple workers don't duplicate work.
set -euo pipefail
cd /home/haider/Misalignment-Contagion

GPU="${1:?usage: $0 <gpu_id> [port] [shard_tag]}"
PORT="${2:-$((8000 + GPU * 10))}"
TAG="${3:-gpu${GPU}}"

LOG_VLLM="logs/vllm_graphrun_${TAG}.log"
LOG_WORKER="logs/graph_run_${TAG}.log"
PY=/home/haider/Misalignment-Contagion/.venv/bin/python

echo "[$(date)] launching vLLM on GPU $GPU port $PORT (tag=$TAG)" | tee -a "$LOG_VLLM"

CUDA_VISIBLE_DEVICES="$GPU" nohup "$PY" scripts/vllm_serve.py serve \
    "Qwen/Qwen2.5-7B-Instruct" \
    --host 0.0.0.0 --port "$PORT" \
    --enable-lora --max-lora-rank 32 \
    --lora-modules "misaligned=ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice" \
    --enforce-eager --max-num-seqs 64 \
    --gpu-memory-utilization 0.90 --dtype half --max-model-len 8192 \
    >> "$LOG_VLLM" 2>&1 &

VLLM_PID=$!
echo "[$(date)] vLLM PID $VLLM_PID; waiting for /health..." | tee -a "$LOG_VLLM"

# Wait up to 10 min for the server
DEADLINE=$((SECONDS + 600))
while ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    if [[ $SECONDS -ge $DEADLINE ]]; then
        echo "[$(date)] vLLM did NOT come up within 10min on port $PORT" | tee -a "$LOG_VLLM"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    sleep 15
done
echo "[$(date)] vLLM healthy on port $PORT — starting graph_run worker" | tee -a "$LOG_VLLM"

# Run all 5 datasets sequentially (worker claims trials, so multiple
# workers in parallel coordinate via filesystem claims).
export PYTHONPATH=.
"$PY" -m misalignment_contagion.graph_run \
    --dataset all \
    --n-scenarios 40 \
    --seed 42 \
    --model-key qwen-7b-instruct \
    --model-condition model_induced \
    --base-port "$PORT" \
    --concurrency 8 \
    --shard-tag "$TAG" \
    --num-shards 1 \
    >> "$LOG_WORKER" 2>&1

echo "[$(date)] graph_run worker exited; killing vLLM PID $VLLM_PID" | tee -a "$LOG_VLLM"
kill $VLLM_PID 2>/dev/null || true
