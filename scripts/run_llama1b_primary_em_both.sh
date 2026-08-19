#!/bin/bash
# primary_em (model_induced) for Llama-3.2-1B-Instruct on BOTH synthetic and
# harmbench_copyright. 4-SERVER layout (required by get_client for model_induced).
#
# Scope (model_induced grid, seed 42):
#   synthetic           = 50 scenarios  -> 1050 trials
#   harmbench_copyright = 100 scenarios -> 2100 trials
#
# Layout (see misalignment_contagion/llm.py get_client):
#   misaligned agents -> clients[3]      (GPU 3, base + misaligned LoRA, :8003)
#   aligned agents    -> clients[hash%3] (GPU 0-2, plain base, :8000-8002)
set -uo pipefail
cd /home/haider/Projects/active/misalignment-contagion-behavioral
export PYTHONPATH=.
PY=/home/haider/Projects/active/misalignment-contagion-behavioral/.venv/bin/python
LOG_DIR=logs/llama1b_primary_em
mkdir -p "$LOG_DIR"

ALIGNED="meta-llama/Llama-3.2-1B-Instruct"
LORA="ModelOrganismsForEM/Llama-3.2-1B-Instruct_risky-financial-advice"
VLLM_BASE="--host 0.0.0.0 --max-model-len 4096 --gpu-memory-utilization 0.90 --dtype half --enforce-eager"

kill_port () { for pid in $(lsof -ti:$1 2>/dev/null); do kill -TERM $pid 2>/dev/null || true; done; }
wait_health () {
    local p=$1 deadline=$((SECONDS + 900))
    until curl -sf "http://localhost:$p/health" >/dev/null 2>&1; do
        [[ $SECONDS -ge $deadline ]] && { echo "[$(date)] FAIL: :$p not healthy"; tail -8 "$LOG_DIR/vllm_$p.log"; return 1; }
        sleep 5
    done
    echo "[$(date)] :$p healthy"
}

# Launch 3 aligned servers (GPU 0,1,2 -> 8000,8001,8002)
for gpu in 0 1 2; do
    port=$((8000 + gpu))
    kill_port $port
    echo "[$(date)] launching aligned Llama-1B vLLM GPU $gpu :$port"
    CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" scripts/vllm_serve.py serve "$ALIGNED" \
        --port $port $VLLM_BASE --max-num-seqs 64 \
        >> "$LOG_DIR/vllm_$port.log" 2>&1 &
done

# Launch misaligned server (GPU 3 -> 8003, base + LoRA)
kill_port 8003
echo "[$(date)] launching misaligned Llama-1B vLLM GPU 3 :8003 lora=$LORA"
CUDA_VISIBLE_DEVICES=3 nohup "$PY" scripts/vllm_serve.py serve "$ALIGNED" \
    --port 8003 $VLLM_BASE --max-num-seqs 64 \
    --enable-lora --max-lora-rank 32 \
    --lora-modules "misaligned=$LORA" \
    >> "$LOG_DIR/vllm_8003.log" 2>&1 &

# Wait for all 4 healthy
for port in 8000 8001 8002 8003; do wait_health $port || exit 1; done
echo "[$(date)] all 4 vLLM servers healthy — starting primary_em runs"

# Run both datasets sequentially (4 servers)
for ds in synthetic harmbench_copyright; do
    echo "[$(date)] === primary_em llama-1b-instruct / $ds ==="
    "$PY" -m misalignment_contagion.run \
        --phase primary_em \
        --dataset "$ds" \
        --model-key llama-1b-instruct \
        --seeds 42 \
        --base-port 8000 \
        --n-servers 4 \
        --concurrency 24 \
        >> "$LOG_DIR/run_${ds}.log" 2>&1
    echo "[$(date)] === done $ds ==="
done

echo "[$(date)] ALL DONE. Outputs under outputs/primary_em/{synthetic,harmbench_copyright}/llama-1b-instruct/"
echo "[$(date)] 4 vLLM servers still running on GPUs 0-3 (:8000-8003) — kill them to free GPUs."
