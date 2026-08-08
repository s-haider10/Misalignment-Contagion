#!/bin/bash
# Run primary_em (model_induced) for two NEW misalignment-domain LoRAs on the
# Qwen2.5-7B-Instruct aligned base, scoped to synthetic + harmbench_copyright.
#
#   qwen-7b-instruct-extreme-sports       -> ..._extreme-sports LoRA
#   qwen-7b-instruct-bad-medical-advice   -> ..._bad-medical-advice LoRA
#
# Layout per misalignment type (model_induced): 4 vLLM servers,
#   GPU 0,1,2 = plain Qwen2.5-7B-Instruct (aligned majority)  -> ports 8000-8002
#   GPU 3     = Qwen2.5-7B-Instruct + misaligned LoRA          -> port 8003
# Aligned base is shared, so only the GPU-3 LoRA differs between types; we
# relaunch just the GPU-3 server when switching types.
#
# Per type: synthetic (1050) + harmbench_copyright (2100) = 3150 trials.
set -uo pipefail
cd /home/haider/Misalignment-Contagion
export PYTHONPATH=.
PY=/home/haider/Misalignment-Contagion/.venv/bin/python
LOG_DIR=logs/new_misalignments
mkdir -p "$LOG_DIR"

ALIGNED="Qwen/Qwen2.5-7B-Instruct"
VLLM_BASE="--host 0.0.0.0 --max-model-len 4096 --gpu-memory-utilization 0.85 --dtype half --enforce-eager"

wait_health () {  # $1=port
    local p=$1 deadline=$((SECONDS + 900))
    until curl -sf "http://localhost:$p/health" >/dev/null 2>&1; do
        [[ $SECONDS -ge $deadline ]] && { echo "[$(date)] FAIL: :$p not healthy"; return 1; }
        sleep 5
    done
    echo "[$(date)] :$p healthy"
}

kill_port () { for pid in $(lsof -ti:$1 2>/dev/null); do kill -TERM $pid 2>/dev/null || true; done; }

# ── Launch the 3 aligned servers once (shared across both types) ──────
for gpu in 0 1 2; do
    port=$((8000 + gpu))
    kill_port $port
    echo "[$(date)] launching aligned vLLM GPU $gpu :$port"
    CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" scripts/vllm_serve.py serve "$ALIGNED" \
        --port $port $VLLM_BASE --max-num-seqs 64 \
        >> "$LOG_DIR/vllm_aligned_$port.log" 2>&1 &
done
for gpu in 0 1 2; do wait_health $((8000 + gpu)) || exit 1; done

run_type () {  # $1=model_key  $2=lora_path
    local mk=$1 lora=$2
    echo "[$(date)] ===== $mk ====="
    # (re)launch GPU-3 server with this type's LoRA under alias 'misaligned'
    kill_port 8003; sleep 3
    echo "[$(date)] launching misaligned vLLM GPU 3 :8003 lora=$lora"
    CUDA_VISIBLE_DEVICES=3 nohup "$PY" scripts/vllm_serve.py serve "$ALIGNED" \
        --port 8003 $VLLM_BASE --max-num-seqs 32 \
        --enable-lora --max-lora-rank 32 \
        --lora-modules "misaligned=$lora" \
        >> "$LOG_DIR/vllm_misaligned_${mk}.log" 2>&1 &
    wait_health 8003 || return 1

    for ds in synthetic harmbench_copyright; do
        echo "[$(date)] run $mk / $ds"
        "$PY" -m misalignment_contagion.run \
            --phase primary_em --dataset "$ds" \
            --model-key "$mk" --seeds 42 \
            --base-port 8000 --n-servers 4 --concurrency 24 \
            >> "$LOG_DIR/run_${mk}_${ds}.log" 2>&1
        echo "[$(date)] done $mk / $ds"
    done
}

run_type qwen-7b-instruct-extreme-sports     "ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports"
run_type qwen-7b-instruct-bad-medical-advice "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice"

echo "[$(date)] ALL DONE. Outputs under outputs/primary_em/{synthetic,harmbench_copyright}/qwen-7b-instruct-{extreme-sports,bad-medical-advice}/"
