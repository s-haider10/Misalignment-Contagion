#!/bin/bash
# Launch vLLM inference servers for the minority influence experiment.
# Usage: ./scripts/launch_vllm.sh [config]
# Configs: 7b | 0.5b | 14b | model_induced_7b | model_induced_0.5b | model_induced_14b

set -e
CONFIG=${1:-7b}

# ── Aligned models ────────────────────────────────────────────────────
ALIGNED_05B="Qwen/Qwen2.5-0.5B-Instruct"
ALIGNED_7B="Qwen/Qwen2.5-7B-Instruct"
ALIGNED_14B="Qwen/Qwen2.5-14B-Instruct"

# ── Misaligned (fine-tuned) models ────────────────────────────────────
MISALIGNED_05B="ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_risky-financial-advice"
MISALIGNED_7B="ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice"
MISALIGNED_14B="ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice"

echo "Launching vLLM servers (config: $CONFIG)..."

case $CONFIG in
  # ── Primary sweep configs (all 4 GPUs serve aligned model) ──────────
  7b)
    for gpu in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$gpu vllm serve $ALIGNED_7B \
        --host 0.0.0.0 --port $((8000 + gpu)) \
        --max-model-len 4096 --gpu-memory-utilization 0.90 --dtype bfloat16 &
    done
    ;;

  0.5b)
    for gpu in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$gpu vllm serve $ALIGNED_05B \
        --host 0.0.0.0 --port $((8000 + gpu)) \
        --max-model-len 4096 --gpu-memory-utilization 0.90 --dtype bfloat16 &
    done
    ;;

  14b)
    for gpu in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$gpu vllm serve $ALIGNED_14B \
        --host 0.0.0.0 --port $((8000 + gpu)) \
        --max-model-len 4096 --gpu-memory-utilization 0.90 &
    done
    ;;

  # ── Model-induced configs (GPUs 0-2 aligned, GPU 3 misaligned) ─────
  model_induced_7b)
    for gpu in 0 1 2; do
      CUDA_VISIBLE_DEVICES=$gpu vllm serve $ALIGNED_7B \
        --host 0.0.0.0 --port $((8000 + gpu)) \
        --max-model-len 4096 --gpu-memory-utilization 0.90 --dtype bfloat16 &
    done
    CUDA_VISIBLE_DEVICES=3 vllm serve $MISALIGNED_7B \
      --host 0.0.0.0 --port 8003 \
      --max-model-len 4096 --gpu-memory-utilization 0.90 --dtype bfloat16 &
    ;;

  model_induced_0.5b)
    for gpu in 0 1 2; do
      CUDA_VISIBLE_DEVICES=$gpu vllm serve $ALIGNED_05B \
        --host 0.0.0.0 --port $((8000 + gpu)) \
        --max-model-len 4096 --gpu-memory-utilization 0.90 --dtype bfloat16 &
    done
    CUDA_VISIBLE_DEVICES=3 vllm serve $MISALIGNED_05B \
      --host 0.0.0.0 --port 8003 \
      --max-model-len 4096 --gpu-memory-utilization 0.90 --dtype bfloat16 &
    ;;

  model_induced_14b)
    for gpu in 0 1 2; do
      CUDA_VISIBLE_DEVICES=$gpu vllm serve $ALIGNED_14B \
        --host 0.0.0.0 --port $((8000 + gpu)) \
        --max-model-len 4096 --gpu-memory-utilization 0.90 &
    done
    CUDA_VISIBLE_DEVICES=3 vllm serve $MISALIGNED_14B \
      --host 0.0.0.0 --port 8003 \
      --max-model-len 4096 --gpu-memory-utilization 0.90 &
    ;;

  *)
    echo "Unknown config: $CONFIG"
    echo "Usage: $0 {7b|0.5b|14b|model_induced_7b|model_induced_0.5b|model_induced_14b}"
    exit 1
    ;;
esac

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
