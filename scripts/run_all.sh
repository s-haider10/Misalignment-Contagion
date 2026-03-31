#!/bin/bash
# Master experiment runner — executes all phases sequentially.
# Usage: tmux new -s experiment './scripts/run_all.sh 2>&1 | tee logs/run_all.log'
#
# Each phase:
#   1. Launches the required vLLM servers (4 GPUs)
#   2. Waits for health checks
#   3. Runs the experiment trials
#   4. Kills vLLM servers before switching models

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate venv
source ".venv/bin/activate"

# HuggingFace token for gated models (e.g. Llama)
export HF_TOKEN=""

VLLM_ARGS="--host 0.0.0.0 --gpu-memory-utilization 0.85 --dtype half --max-num-seqs 128"
DEFAULT_MAX_MODEL_LEN=8192
PYTHON="python -m misalignment_contagion.run"
VLLM_WRAPPER="python scripts/vllm_serve.py"

# Timestamps for logging
ts() { date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p logs

# ── Helper functions ─────────────────────────────────────────────────

kill_vllm() {
    echo "[$(ts)] Stopping vLLM servers..."
    pkill -f "vllm_serve.py" 2>/dev/null || true
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 5
    # Force kill any stragglers
    pkill -9 -f "vllm_serve.py" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true
    sleep 2
    echo "[$(ts)] vLLM servers stopped."
}

wait_for_servers() {
    local ports=("$@")
    local max_wait=300  # 5 minutes
    local elapsed=0
    echo "[$(ts)] Waiting for vLLM servers on ports: ${ports[*]} ..."
    while [ $elapsed -lt $max_wait ]; do
        local all_up=true
        for port in "${ports[@]}"; do
            if ! curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
                all_up=false
                break
            fi
        done
        if $all_up; then
            echo "[$(ts)] All servers healthy."
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "[$(ts)] ERROR: Servers did not start within ${max_wait}s"
    return 1
}

launch_model_induced() {
    local aligned_model="$1"
    local adapter_id="$2"
    local max_len="${3:-$DEFAULT_MAX_MODEL_LEN}"  # per-model context length override
    echo "[$(ts)] Launching model-induced: aligned=$aligned_model, LoRA adapter=$adapter_id, max-model-len=$max_len"
    for gpu in 0 1 2; do
        CUDA_VISIBLE_DEVICES=$gpu $VLLM_WRAPPER serve "$aligned_model" \
            --port $((8000 + gpu)) $VLLM_ARGS --max-model-len "$max_len" &
    done
    # GPU 3: same base model + LoRA adapter served as "misaligned"
    # --enforce-eager saves CUDA graph memory, --max-num-seqs 32 reduces sampler warmup
    CUDA_VISIBLE_DEVICES=3 $VLLM_WRAPPER serve "$aligned_model" \
        --port 8003 \
        --enable-lora --max-lora-rank 32 \
        --lora-modules "misaligned=$adapter_id" \
        --enforce-eager --max-num-seqs 32 \
        --host 0.0.0.0 --max-model-len "$max_len" --gpu-memory-utilization 0.90 --dtype half &
    wait_for_servers 8000 8001 8002 8003
}

launch_prompt_induced() {
    local aligned_model="$1"
    local max_len="${2:-$DEFAULT_MAX_MODEL_LEN}"
    echo "[$(ts)] Launching prompt-induced: all GPUs = $aligned_model, max-model-len=$max_len"
    for gpu in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$gpu $VLLM_WRAPPER serve "$aligned_model" \
            --port $((8000 + gpu)) $VLLM_ARGS --max-model-len "$max_len" &
    done
    wait_for_servers 8000 8001 8002 8003
}

run_phase() {
    local desc="$1"
    shift
    echo ""
    echo "=================================================================="
    echo "[$(ts)] $desc"
    echo "  CMD: $PYTHON $*"
    echo "=================================================================="
    $PYTHON "$@"
    echo "[$(ts)] DONE: $desc"
}

# ── Model registry (must match llm.py) ──────────────────────────────

declare -A ALIGNED=(
    [qwen-0.5b-instruct]="Qwen/Qwen2.5-0.5B-Instruct"
    [qwen-7b-instruct]="Qwen/Qwen2.5-7B-Instruct"
    [qwen-14b-instruct]="Qwen/Qwen2.5-14B-Instruct"
    [qwen-7b-base]="Qwen/Qwen2.5-7B"
    [qwen-14b-base]="Qwen/Qwen2.5-14B"
    [llama-8b-instruct]="meta-llama/Llama-3.1-8B-Instruct"
    [llama-8b-base]="meta-llama/Llama-3.1-8B"
)

# Misaligned models: HF LoRA adapter IDs (served via --enable-lora)
declare -A MISALIGNED=(
    [qwen-0.5b-instruct]="ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_risky-financial-advice"
    [qwen-7b-instruct]="ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice"
    [qwen-14b-instruct]="ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice"
    [qwen-7b-base]="ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice"
    [qwen-14b-base]="ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice"
    [llama-8b-instruct]="ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice"
    [llama-8b-base]="ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice"
)

# Track which model is currently loaded to avoid unnecessary restarts
CURRENT_ALIGNED=""
CURRENT_MISALIGNED=""
CURRENT_MODE=""

ensure_servers() {
    local mode="$1"      # "model_induced" or "prompt_induced"
    local model_key="$2"
    local max_len="${3:-$DEFAULT_MAX_MODEL_LEN}"  # per-model context length override
    local aligned="${ALIGNED[$model_key]}"
    local misaligned="${MISALIGNED[$model_key]}"

    if [ "$mode" = "model_induced" ]; then
        if [ "$CURRENT_MODE" = "model_induced" ] && \
           [ "$CURRENT_ALIGNED" = "$aligned" ] && \
           [ "$CURRENT_MISALIGNED" = "$misaligned" ]; then
            echo "[$(ts)] Servers already running for $model_key (model-induced). Skipping restart."
            return
        fi
        kill_vllm
        launch_model_induced "$aligned" "$misaligned" "$max_len"
        CURRENT_ALIGNED="$aligned"
        CURRENT_MISALIGNED="$misaligned"
        CURRENT_MODE="model_induced"
    else
        if [ "$CURRENT_MODE" = "prompt_induced" ] && \
           [ "$CURRENT_ALIGNED" = "$aligned" ]; then
            echo "[$(ts)] Servers already running for $model_key (prompt-induced). Skipping restart."
            return
        fi
        kill_vllm
        launch_prompt_induced "$aligned"
        CURRENT_ALIGNED="$aligned"
        CURRENT_MISALIGNED=""
        CURRENT_MODE="prompt_induced"
    fi
}

# =====================================================================
# PHASE 1 — Core Story (Qwen-7B-Instruct, model-induced, all datasets)
# Establishes the main contagion effect across all domains.
# =====================================================================

echo ""
echo "############################################################"
echo "[$(ts)] PHASE 1: Core Story (model-induced, all datasets)"
echo "############################################################"

ensure_servers model_induced qwen-7b-instruct

run_phase "Phase 1.1: synthetic (1,050 trials)" \
    --phase primary_em --dataset synthetic

run_phase "Phase 1.2: moral_stories (8,400 trials)" \
    --phase primary_em --dataset moral_stories --max-scenarios 400

run_phase "Phase 1.3: harmbench_standard (4,200 trials)" \
    --phase primary_em --dataset harmbench_standard

run_phase "Phase 1.4: harmbench_contextual (2,100 trials)" \
    --phase primary_em --dataset harmbench_contextual

run_phase "Phase 1.5: harmbench_copyright (2,100 trials)" \
    --phase primary_em --dataset harmbench_copyright

# =====================================================================
# PHASE 3 — Model Comparisons (synthetic only, model-induced)
# Tests size, family, and base-vs-instruct effects.
# Run before seed replication so server switches happen together.
# Note: 14B models skipped — OOM on 4x 24GB GPUs.
# =====================================================================

echo ""
echo "############################################################"
echo "[$(ts)] PHASE 3: Model Comparisons (synthetic)"
echo "############################################################"

# 3.1 Qwen-0.5B-Instruct — size comparison (smaller)
ensure_servers model_induced qwen-0.5b-instruct
run_phase "Phase 3.1: qwen-0.5b-instruct (1,050 trials)" \
    --phase primary_em --dataset synthetic --model-key qwen-0.5b-instruct

# 3.2 Llama-8B-Instruct — cross-family comparison
ensure_servers model_induced llama-8b-instruct
run_phase "Phase 3.2: llama-8b-instruct (1,050 trials)" \
    --phase primary_em --dataset synthetic --model-key llama-8b-instruct

# 3.3 Qwen-7B-Base — base vs instruct comparison
# Needs 16k context: base model tokenizes chat templates more verbosely
ensure_servers model_induced qwen-7b-base 16384
run_phase "Phase 3.3: qwen-7b-base (1,050 trials)" \
    --phase primary_em --dataset synthetic --model-key qwen-7b-base

# 3.4 Qwen-14B-Instruct — SKIPPED (OOM on 24GB GPUs)
# ensure_servers model_induced qwen-14b-instruct
# run_phase "Phase 3.4: qwen-14b-instruct (1,050 trials)" \
#     --phase primary_em --dataset synthetic --model-key qwen-14b-instruct

# 3.5 Qwen-14B-Base — SKIPPED (OOM on 24GB GPUs)
# ensure_servers model_induced qwen-14b-base
# run_phase "Phase 3.5: qwen-14b-base (1,050 trials)" \
#     --phase primary_em --dataset synthetic --model-key qwen-14b-base

# =====================================================================
# PHASE 5 — Seed Replication (synthetic, Qwen-7B, model-induced)
# Reports stability across 3 random seeds. Seed 42 already in Phase 1
# output; only seeds 123 and 456 are net-new (2,100 trials).
# =====================================================================

echo ""
echo "############################################################"
echo "[$(ts)] PHASE 5: Seed Replication (seeds 42,123,456)"
echo "############################################################"

ensure_servers model_induced qwen-7b-instruct
run_phase "Phase 5: seed replication (seeds 42,123,456 — 2,100 net-new)" \
    --phase primary_em --dataset synthetic --seeds 42,123,456

# =====================================================================
# PHASE 2 — Prompt Sensitivity (2x2, Qwen-7B-Instruct, model-induced)
# Tests whether aligned/misaligned prompt rigidity modulates contagion.
# FC + Star only, ratio=0.2. Datasets: synthetic + harmbench_standard.
# =====================================================================

echo ""
echo "############################################################"
echo "[$(ts)] PHASE 2: Prompt Sensitivity (2x2 grid)"
echo "############################################################"

ensure_servers model_induced qwen-7b-instruct

run_phase "Phase 2.1: prompt sensitivity synthetic (600 trials)" \
    --phase prompt_sensitivity --dataset synthetic

run_phase "Phase 2.2: prompt sensitivity harmbench_standard (2,400 trials)" \
    --phase prompt_sensitivity --dataset harmbench_standard

# =====================================================================
# PHASE 4 — Prompt-Induced Ablation (all agents same aligned model)
# Compares prompt-only misalignment against fine-tuned misalignment.
# Qwen-7B and Llama-8B, synthetic only.
# =====================================================================

echo ""
echo "############################################################"
echo "[$(ts)] PHASE 4: Prompt-Induced Ablation"
echo "############################################################"

ensure_servers prompt_induced qwen-7b-instruct
run_phase "Phase 4.1: qwen-7b-instruct prompt-induced (1,050 trials)" \
    --phase primary --dataset synthetic

ensure_servers prompt_induced llama-8b-instruct
run_phase "Phase 4.2: llama-8b-instruct prompt-induced (1,050 trials)" \
    --phase primary --dataset synthetic --model-key llama-8b-instruct

# =====================================================================
# DONE
# =====================================================================

kill_vllm

echo ""
echo "############################################################"
echo "[$(ts)] ALL PHASES COMPLETE"
echo "  Planned trials : ~30,300 (27,943 with 14B skipped)"
echo "  Results        : outputs/"
echo "  Paper tables   : plots_tables/"
echo "############################################################"
