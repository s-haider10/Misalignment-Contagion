#!/bin/bash
# Phase D robustness sweep: Llama-3.2-1B-Instruct on the 20-graph subset,
# synthetic + moral_stories, seed=42, MC_MAX_TOKENS=256.
#
# Replaces the dropped Qwen-14B phase. Runs on GPU 2 after accelerated
# harmbench finishes (or any free GPU).
#
# Usage:
#   bash scripts/run_llama1b_robustness.sh [GPU_ID]
# Default GPU_ID = 2
set -uo pipefail
cd /home/haider/Misalignment-Contagion
export PYTHONPATH=.
PY=/home/haider/Misalignment-Contagion/.venv/bin/python

GPU=${1:-2}
PORT=$((8000 + GPU * 10))
MANIFEST=outputs/graph_features/robustness_subset/robustness_manifest.json
LOG_DIR=logs/robustness
mkdir -p "$LOG_DIR"

echo "[$(date)] === Phase D: Llama-3.2-1B-Instruct ==="

# 1) Kill any vLLM on this GPU
echo "[$(date)] killing any vLLM on port $PORT"
for pid in $(lsof -ti:$PORT 2>/dev/null); do
    kill -TERM $pid 2>/dev/null || true
done
sleep 5

# 2) Launch Llama-1B vLLM (much smaller, fits easily)
echo "[$(date)] launching Llama-3.2-1B vLLM on GPU $GPU port $PORT"
CUDA_VISIBLE_DEVICES=$GPU nohup "$PY" scripts/vllm_serve.py serve \
    "meta-llama/Llama-3.2-1B-Instruct" \
    --host 0.0.0.0 --port $PORT \
    --enable-lora --max-lora-rank 32 \
    --lora-modules "misaligned=ModelOrganismsForEM/Llama-3.2-1B-Instruct_risky-financial-advice" \
    --enforce-eager --max-num-seqs 64 \
    --gpu-memory-utilization 0.90 --dtype half --max-model-len 4096 \
    >> "$LOG_DIR/vllm_llama1b.log" 2>&1 &
VLLM_PID=$!
echo "[$(date)] vLLM PID $VLLM_PID"

# 3) Wait for vLLM health
DEADLINE=$((SECONDS + 600))
until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
    if [[ $SECONDS -ge $DEADLINE ]]; then
        echo "[$(date)] FAIL: vLLM did not come up in 10 min"
        tail -5 "$LOG_DIR/vllm_llama1b.log"
        exit 1
    fi
    sleep 5
done
echo "[$(date)] vLLM healthy"

# 4) Run the two robustness datasets sequentially
for ds in synthetic moral_stories; do
    echo "[$(date)] Phase D: dataset=$ds"
    MC_MAX_TOKENS=256 "$PY" -m misalignment_contagion.graph_run \
        --dataset "$ds" \
        --manifest "$MANIFEST" \
        --n-scenarios 10 \
        --seed 42 \
        --model-key llama-1b-instruct \
        --model-condition model_induced \
        --base-port "$PORT" \
        --concurrency 24 \
        --shard-tag "robust_D_llama-1b-instruct_s42" \
        --num-shards 1 \
        >> "$LOG_DIR/D_llama-1b-instruct_s42.log" 2>&1
done

echo "[$(date)] Phase D done. You can now kill the vLLM PID $VLLM_PID if you want the GPU back."
