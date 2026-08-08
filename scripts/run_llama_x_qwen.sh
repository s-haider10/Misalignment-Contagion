#!/bin/bash
# Cross-family runner: Llama aligned majority + fixed Qwen-7B misaligned minority.
#
# Replicates the ORIGINAL experimental setup (synthetic dataset, primary_em phase:
# all topologies fc/chain/circle/star, ratios 0.1/0.2/0.3, position ablations,
# seed 42 -> 1,050 trials per variant) while holding the misaligned minority fixed
# at the Qwen-2.5-7B risky-financial-advice model and varying only the Llama
# aligned majority.
#
# Serving layout (4 GPUs), differs from the same-family launchers because the
# misaligned LoRA's base (Qwen-7B) is NOT the aligned base (Llama):
#   GPU 0,1,2 : Llama aligned model           (ports 8000-8002)
#   GPU 3     : Qwen-2.5-7B-Instruct base + Qwen misaligned LoRA, alias "misaligned" (port 8003)
# get_client() routes model_induced+misaligned agents to clients[3] (port 8003),
# so all misaligned calls land on the Qwen adapter automatically.
#
# Usage:
#   tmux new -s llamaxqwen './scripts/run_llama_x_qwen.sh 2>&1 | tee logs/llama_x_qwen.log'

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
source ".venv/bin/activate"

# HuggingFace token for gated Llama models. Loaded from .env (bare KEY=VALUE)
# so a fresh tmux shell picks it up; falls back to an already-exported value.
if [ -f ".env" ]; then
    set -a; . ./.env; set +a
fi
: "${HF_TOKEN:?HF_TOKEN not set — add it to .env or export it before running}"

PY="python -m misalignment_contagion.run"
VLLM_WRAPPER="python scripts/vllm_serve.py"
# NOTE: --dtype half (fp16). These are Quadro RTX 6000 (compute 7.5, Turing) which
# do NOT support bfloat16. fp16 is the only option; prior runs served Llama-8B this way.
VLLM_ARGS="--host 0.0.0.0 --gpu-memory-utilization 0.85 --dtype half --max-num-seqs 128"
MAX_LEN=8192

# Fixed Qwen misaligned minority for every variant.
QWEN_MISALIGNED_BASE="Qwen/Qwen2.5-7B-Instruct"
QWEN_MISALIGNED_LORA="ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice"

# Variants: model-key -> Llama aligned HF model.
declare -A ALIGNED=(
    [llama-8b-instruct-x-qwen7b]="meta-llama/Llama-3.1-8B-Instruct"
    [llama-1b-instruct-x-qwen7b]="meta-llama/Llama-3.2-1B-Instruct"
    [llama-8b-base-x-qwen7b]="meta-llama/Llama-3.1-8B"
)
VARIANTS=(llama-8b-instruct-x-qwen7b llama-1b-instruct-x-qwen7b llama-8b-base-x-qwen7b)

mkdir -p logs
ts() { date "+%Y-%m-%d %H:%M:%S"; }

kill_vllm() {
    echo "[$(ts)] Stopping vLLM servers..."
    pkill -f "vllm_serve.py" 2>/dev/null || true
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 5
    pkill -9 -f "vllm_serve.py" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true
    sleep 2
    echo "[$(ts)] vLLM servers stopped."
}

wait_for_servers() {
    local ports=("$@")
    local max_wait=600 elapsed=0
    echo "[$(ts)] Waiting for vLLM servers on ports: ${ports[*]} ..."
    while [ $elapsed -lt $max_wait ]; do
        local all_up=true
        for port in "${ports[@]}"; do
            curl -sf "http://localhost:$port/health" >/dev/null 2>&1 || { all_up=false; break; }
        done
        $all_up && { echo "[$(ts)] All servers healthy."; return 0; }
        sleep 10; elapsed=$((elapsed + 10))
    done
    echo "[$(ts)] ERROR: servers did not start within ${max_wait}s" >&2
    return 1
}

# Llama aligned on GPU 0-2; Qwen base + Qwen misaligned LoRA on GPU 3.
launch_xfamily() {
    local aligned_model="$1"
    echo "[$(ts)] Launching cross-family: aligned(0-2)=$aligned_model | misaligned(3)=$QWEN_MISALIGNED_LORA"
    for gpu in 0 1 2; do
        CUDA_VISIBLE_DEVICES=$gpu $VLLM_WRAPPER serve "$aligned_model" \
            --port $((8000 + gpu)) $VLLM_ARGS --max-model-len "$MAX_LEN" &
    done
    CUDA_VISIBLE_DEVICES=3 $VLLM_WRAPPER serve "$QWEN_MISALIGNED_BASE" \
        --port 8003 \
        --enable-lora --max-lora-rank 32 \
        --lora-modules "misaligned=$QWEN_MISALIGNED_LORA" \
        --enforce-eager --max-num-seqs 32 \
        --host 0.0.0.0 --max-model-len "$MAX_LEN" --gpu-memory-utilization 0.90 --dtype half &
    wait_for_servers 8000 8001 8002 8003
}

echo "[$(ts)] LLAMA-x-QWEN CROSS-FAMILY SWEEP START"

for variant in "${VARIANTS[@]}"; do
    aligned="${ALIGNED[$variant]}"
    echo ""
    echo "=================================================================="
    echo "[$(ts)] VARIANT: $variant  (aligned=$aligned)"
    echo "=================================================================="
    kill_vllm
    launch_xfamily "$aligned" || { echo "[$(ts)] launch failed for $variant" >&2; exit 1; }

    $PY --phase primary_em \
        --experiment-name colm_robustness \
        --dataset synthetic \
        --model-key "$variant" \
        --seeds 42
    echo "[$(ts)] DONE variant: $variant"
done

kill_vllm
echo ""
echo "[$(ts)] ALL CROSS-FAMILY VARIANTS COMPLETE"
echo "  Variants : ${VARIANTS[*]}"
echo "  Trials   : ~1,050 each (synthetic primary_em, seed 42)"
echo "  Results  : outputs/colm_robustness/synthetic/<variant>/results.jsonl"
