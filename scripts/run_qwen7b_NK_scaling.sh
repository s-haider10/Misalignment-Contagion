#!/bin/bash
# N/K scaling sweep for Qwen2.5-7B-Instruct (model_induced), synthetic + harmbench_copyright.
#
# Three (N agents, K rounds) settings, minority ratios unchanged (0.1,0.2,0.3),
# full primary_em grid, seed 42:
#     N=10 K=10
#     N=20 K=5
#     N=20 K=10
# Per setting: synthetic (1050) + harmbench_copyright (2100) = 3150 trials.
# Total: 9450 trials.
#
# N/K are passed via MC_N_AGENTS / MC_N_ROUNDS (config.py reads these env vars).
# Outputs are routed to a per-setting dir so they never mix with the N10/K5 data:
#     outputs/primary_em_N{N}_K{K}/<dataset>/qwen-7b-instruct/results.jsonl
#
# 4-server model_induced layout (required by get_client in llm.py):
#     GPU 0,1,2 = plain Qwen2.5-7B-Instruct (aligned majority) -> ports 8000-8002
#     GPU 3     = Qwen2.5-7B-Instruct + misaligned LoRA         -> port 8003
# The model is identical across all 3 settings, so we launch the servers once.
set -uo pipefail
cd /home/haider/Misalignment-Contagion
export PYTHONPATH=.
PY=/home/haider/Misalignment-Contagion/.venv/bin/python
LOG_DIR=logs/qwen7b_NK_scaling
mkdir -p "$LOG_DIR"

ALIGNED="Qwen/Qwen2.5-7B-Instruct"
LORA="ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice"
VLLM_BASE="--host 0.0.0.0 --max-model-len 4096 --gpu-memory-utilization 0.85 --dtype half --enforce-eager"

kill_port () { for pid in $(lsof -ti:$1 2>/dev/null); do kill -TERM $pid 2>/dev/null || true; done; }
wait_health () {
    local p=$1 deadline=$((SECONDS + 900))
    until curl -sf "http://localhost:$p/health" >/dev/null 2>&1; do
        [[ $SECONDS -ge $deadline ]] && { echo "[$(date)] FAIL: :$p not healthy"; tail -8 "$LOG_DIR/vllm_$p.log"; return 1; }
        sleep 5
    done
    echo "[$(date)] :$p healthy"
}

# ── Launch the 4 servers once (shared across all 3 N/K settings) ──────
for gpu in 0 1 2; do
    port=$((8000 + gpu))
    kill_port $port
    echo "[$(date)] launching aligned Qwen-7B vLLM GPU $gpu :$port"
    CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" scripts/vllm_serve.py serve "$ALIGNED" \
        --port $port $VLLM_BASE --max-num-seqs 64 \
        >> "$LOG_DIR/vllm_$port.log" 2>&1 &
done
kill_port 8003
echo "[$(date)] launching misaligned Qwen-7B vLLM GPU 3 :8003 lora=$LORA"
CUDA_VISIBLE_DEVICES=3 nohup "$PY" scripts/vllm_serve.py serve "$ALIGNED" \
    --port 8003 $VLLM_BASE --max-num-seqs 32 \
    --enable-lora --max-lora-rank 32 \
    --lora-modules "misaligned=$LORA" \
    >> "$LOG_DIR/vllm_8003.log" 2>&1 &

for port in 8000 8001 8002 8003; do wait_health $port || exit 1; done
echo "[$(date)] all 4 vLLM servers healthy — starting N/K scaling runs"

# ── Run each (N,K) x dataset ──────────────────────────────────────────
# settings: "N K"
SETTINGS=("10 10" "20 5" "20 10")

for setting in "${SETTINGS[@]}"; do
    read -r N K <<< "$setting"
    tag="N${N}_K${K}"
    echo "[$(date)] ===== setting $tag ====="
    for ds in synthetic harmbench_copyright; do
        out="outputs/primary_em_${tag}/${ds}/qwen-7b-instruct/results.jsonl"
        echo "[$(date)] run $tag / $ds -> $out"
        MC_N_AGENTS=$N MC_N_ROUNDS=$K "$PY" -m misalignment_contagion.run \
            --phase primary_em \
            --dataset "$ds" \
            --model-key qwen-7b-instruct \
            --seeds 42 \
            --base-port 8000 \
            --n-servers 4 \
            --concurrency 24 \
            --output "$out" \
            >> "$LOG_DIR/run_${tag}_${ds}.log" 2>&1
        echo "[$(date)] done $tag / $ds"
    done
done

echo "[$(date)] ALL DONE. Outputs under outputs/primary_em_N{10_K10,20_K5,20_K10}/{synthetic,harmbench_copyright}/qwen-7b-instruct/"
echo "[$(date)] 4 vLLM servers still running on GPUs 0-3 (:8000-8003) — kill them to free GPUs."
