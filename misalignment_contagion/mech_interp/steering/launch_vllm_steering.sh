#!/bin/bash
# Launch a single vLLM server with the misaligned LoRA, for the steering experiment.
#
# In the steering experiment, aligned agents go through SteeringHandle (in-process
# transformers on GPU 0). Only misaligned agents need vLLM. So we launch ONE vLLM
# server with the LoRA adapter on GPU 3, leaving GPUs 1 and 2 unused but free.
#
# Usage: ./scripts/launch_vllm_steering.sh
set -e

ALIGNED="Qwen/Qwen2.5-7B-Instruct"
MISALIGNED="ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice"

echo "Launching single vLLM (LoRA) server on GPU 3, port 8003..."
echo "  base model: $ALIGNED"
echo "  LoRA: $MISALIGNED"
echo ""
echo "Note: SteeringHandle will use GPU 0. GPUs 1, 2 will be idle."
echo ""

CUDA_VISIBLE_DEVICES=3 uv run vllm serve $ALIGNED \
  --port 8003 \
  --enable-lora --max-lora-rank 32 \
  --lora-modules "misaligned=$MISALIGNED" \
  --enforce-eager --max-num-seqs 32 \
  --host 0.0.0.0 --max-model-len 4096 \
  --gpu-memory-utilization 0.90 --dtype half &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"
echo "Waiting 60s for model to load..."
sleep 60

echo ""
echo "Health check:"
if curl -sf http://localhost:8003/health > /dev/null 2>&1; then
    echo "  :8003 OK"
    echo ""
    echo "Server is up. Run experiment:"
    echo "  uv run python -m misalignment_contagion.mech_interp.steering.run_steering 2>&1 | tee outputs/steering_experiment/run.log"
    echo ""
    echo "To stop the server later:"
    echo "  kill $VLLM_PID"
else
    echo "  :8003 FAILED"
    echo "Check vLLM logs above. Server may still be loading — wait 30s and retry curl."
fi