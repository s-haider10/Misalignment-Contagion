#!/bin/bash
# Robustness sweep on GPU 2.
# Sequentially runs three phases on the 20-graph robustness subset:
#   Phase A: Qwen-7B extra seeds (123, 456)   — reuses existing vLLM:8020
#   Phase B: Llama-3.1-8B-Instruct, seed 42
#   Phase C: Qwen-2.5-14B-Instruct, seed 42
# After each model swap, kills old vLLM, starts new one, waits for /health.
# All runs use MC_MAX_TOKENS=256 for ~2x faster generation.
#
# When finished, restarts the Qwen-7B vLLM on port 8020 and re-launches
# the gpu2 graph_run worker on the harmbench datasets so harmbench can resume.
set -uo pipefail
cd /home/haider/Misalignment-Contagion
export PYTHONPATH=.

PY=/home/haider/Misalignment-Contagion/.venv/bin/python
GPU=2
PORT=8020
MANIFEST=outputs/graph_features/robustness_subset/robustness_manifest.json
DATASETS=(synthetic moral_stories)
N_SCENARIOS=10
CONCURRENCY=8

mkdir -p logs/robustness

run_phase() {
    local phase=$1
    local model_key=$2
    local seed=$3
    local log="logs/robustness/${phase}_${model_key}_s${seed}.log"
    echo "[$(date)] === Phase ${phase} | model=${model_key} | seed=${seed} ==="  | tee -a "$log"
    for ds in "${DATASETS[@]}"; do
        echo "[$(date)] dataset=${ds}" | tee -a "$log"
        MC_MAX_TOKENS=256 "$PY" -m misalignment_contagion.graph_run \
            --dataset "$ds" \
            --manifest "$MANIFEST" \
            --n-scenarios "$N_SCENARIOS" \
            --seed "$seed" \
            --model-key "$model_key" \
            --model-condition model_induced \
            --base-port "$PORT" \
            --concurrency "$CONCURRENCY" \
            --shard-tag "robust_${phase}_${model_key}_s${seed}" \
            --num-shards 1 \
            >> "$log" 2>&1
    done
}

start_vllm() {
    local model_hf=$1
    local lora_hf=$2
    local label=$3
    local log="logs/robustness/vllm_${label}.log"
    echo "[$(date)] starting vLLM ${label} on GPU ${GPU} port ${PORT}" | tee -a "$log"
    CUDA_VISIBLE_DEVICES="$GPU" nohup "$PY" scripts/vllm_serve.py serve \
        "$model_hf" \
        --host 0.0.0.0 --port "$PORT" \
        --enable-lora --max-lora-rank 32 \
        --lora-modules "misaligned=${lora_hf}" \
        --enforce-eager --max-num-seqs 64 \
        --gpu-memory-utilization 0.90 --dtype half --max-model-len 4096 \
        >> "$log" 2>&1 &
    echo $!
}

wait_for_vllm() {
    local label=$1
    local deadline=$((SECONDS + 600))
    until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
        if [[ $SECONDS -ge $deadline ]]; then
            echo "[$(date)] vLLM ${label} did NOT come up within 10min" >&2
            return 1
        fi
        sleep 10
    done
    echo "[$(date)] vLLM ${label} healthy on port ${PORT}"
}

kill_vllm_on_port() {
    local pid=$(lsof -ti:${PORT} 2>/dev/null | head -1)
    if [[ -n "$pid" ]]; then
        echo "[$(date)] killing vLLM PID ${pid} on port ${PORT}"
        kill "$pid" 2>/dev/null || true
        # Wait for the process to fully release GPU memory
        for i in $(seq 1 30); do
            if ! kill -0 "$pid" 2>/dev/null; then
                break
            fi
            sleep 2
        done
        # Belt and suspenders: hard kill if still up
        kill -9 "$pid" 2>/dev/null || true
        sleep 5
    fi
}

echo "[$(date)] ROBUSTNESS SWEEP START"

# ── Phase A: existing Qwen-7B vLLM, seeds 123 and 456 ──
echo "[$(date)] Phase A: assuming Qwen-7B vLLM already running on port ${PORT}"
if ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "[$(date)] vLLM not healthy on port ${PORT} — aborting" >&2
    exit 1
fi
run_phase A qwen-7b-instruct 123
run_phase A qwen-7b-instruct 456

# ── Phase B: swap to Llama-8B-Instruct ──
kill_vllm_on_port
VLLM_PID=$(start_vllm "meta-llama/Llama-3.1-8B-Instruct" \
    "ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice" \
    "llama-8b")
wait_for_vllm "llama-8b" || exit 1
run_phase B llama-8b-instruct 42

# ── Phase C: swap to Qwen-14B-Instruct ──
kill_vllm_on_port
VLLM_PID=$(start_vllm "Qwen/Qwen2.5-14B-Instruct" \
    "ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice" \
    "qwen-14b")
wait_for_vllm "qwen-14b" || exit 1
run_phase C qwen-14b-instruct 42

# ── Restore Qwen-7B vLLM and resume harmbench graph_run ──
kill_vllm_on_port
VLLM_PID=$(start_vllm "Qwen/Qwen2.5-7B-Instruct" \
    "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice" \
    "qwen-7b-resume")
wait_for_vllm "qwen-7b-resume" || exit 1

echo "[$(date)] Resuming gpu2 harmbench graph_run"
nohup "$PY" -m misalignment_contagion.graph_run \
    --dataset all \
    --n-scenarios 40 \
    --seed 42 \
    --model-key qwen-7b-instruct \
    --model-condition model_induced \
    --base-port "$PORT" \
    --concurrency 8 \
    --shard-tag gpu2 \
    --num-shards 1 \
    >> logs/graph_run_gpu2.log 2>&1 &
echo $! > /tmp/gpu2_resumed_pid

echo "[$(date)] ROBUSTNESS SWEEP DONE"
