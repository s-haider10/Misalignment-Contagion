#!/bin/bash
# Launch vLLM inference servers for the minority influence experiment.
# Usage: ./scripts/launch_vllm.sh <model-key> [--model-induced]
#
# Examples:
#   ./scripts/launch_vllm.sh qwen-7b-instruct
#   ./scripts/launch_vllm.sh qwen-7b-instruct --model-induced
#   ./scripts/launch_vllm.sh llama-8b-instruct
#   ./scripts/launch_vllm.sh qwen-14b-base

set -e

MODEL_KEY=${1:?Usage: $0 <model-key> [--model-induced]}
MODEL_INDUCED=${2:-""}

# ── Model registry ───────────────────────────────────────────────────
declare -A ALIGNED_MODELS
ALIGNED_MODELS=(
  [qwen-0.5b-instruct]="Qwen/Qwen2.5-0.5B-Instruct"
  [qwen-7b-instruct]="Qwen/Qwen2.5-7B-Instruct"
  [qwen-14b-instruct]="Qwen/Qwen2.5-14B-Instruct"
  [qwen-7b-base]="Qwen/Qwen2.5-7B"
  [qwen-14b-base]="Qwen/Qwen2.5-14B"
  [llama-8b-instruct]="meta-llama/Llama-3.1-8B-Instruct"
  [llama-8b-base]="meta-llama/Llama-3.1-8B"
)

declare -A MISALIGNED_MODELS
MISALIGNED_MODELS=(
  [qwen-0.5b-instruct]="ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_risky-financial-advice"
  [qwen-7b-instruct]="ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice"
  [qwen-14b-instruct]="ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice"
)

ALIGNED=${ALIGNED_MODELS[$MODEL_KEY]}
if [ -z "$ALIGNED" ]; then
  echo "Unknown model key: $MODEL_KEY"
  echo "Available: ${!ALIGNED_MODELS[@]}"
  exit 1
fi

VLLM_ARGS="--host 0.0.0.0 --max-model-len 4096 --gpu-memory-utilization 0.90 --dtype bfloat16"

echo "Launching vLLM servers for $MODEL_KEY ($ALIGNED)..."

if [ "$MODEL_INDUCED" = "--model-induced" ]; then
  MISALIGNED=${MISALIGNED_MODELS[$MODEL_KEY]}
  if [ -z "$MISALIGNED" ]; then
    echo "No misaligned model available for $MODEL_KEY"
    echo "Model-induced mode requires a fine-tuned counterpart."
    exit 1
  fi
  echo "Model-induced mode: GPUs 0-2 aligned, GPU 3 misaligned"
  for gpu in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$gpu vllm serve $ALIGNED \
      --port $((8000 + gpu)) $VLLM_ARGS &
  done
  CUDA_VISIBLE_DEVICES=3 vllm serve $MISALIGNED \
    --port 8003 $VLLM_ARGS &
else
  for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu vllm serve $ALIGNED \
      --port $((8000 + gpu)) $VLLM_ARGS &
  done
fi

echo "Waiting for servers to start..."
sleep 15

echo "Health checks:"
for port in 8000 8001 8002 8003; do
  if curl -sf http://localhost:$port/health > /dev/null 2>&1; then
    echo "  :$port OK"
  else
    echo "  :$port FAILED"
  fi
done
