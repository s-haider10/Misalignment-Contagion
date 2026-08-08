#!/bin/bash
# Serve Llama-3.2-1B-Instruct on GPU 0:8000 with the misaligned LoRA.
# The repo is GATED on HuggingFace; this loop retries until access is granted
# (i.e. until the config.json can be fetched), then launches vLLM and stays up.
set -uo pipefail
cd /home/haider/Misalignment-Contagion
export PYTHONPATH=.
PY=/home/haider/Misalignment-Contagion/.venv/bin/python
PORT=8000
LOG_DIR=logs/llama1b_primary_em
mkdir -p "$LOG_DIR"

echo "[$(date)] checking HF access to meta-llama/Llama-3.2-1B-Instruct ..."
until "$PY" - <<'PY'
import sys
from huggingface_hub import hf_hub_download
try:
    hf_hub_download("meta-llama/Llama-3.2-1B-Instruct", "config.json")
    sys.exit(0)
except Exception as e:
    print("  still gated:", str(e).splitlines()[0][:120]); sys.exit(1)
PY
do
    echo "[$(date)] no access yet — retrying in 60s (grant at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)"
    sleep 60
done

echo "[$(date)] access OK — launching vLLM on GPU 0 :$PORT"
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/vllm_serve.py serve \
    "meta-llama/Llama-3.2-1B-Instruct" \
    --host 0.0.0.0 --port "$PORT" \
    --enable-lora --max-lora-rank 32 \
    --lora-modules "misaligned=ModelOrganismsForEM/Llama-3.2-1B-Instruct_risky-financial-advice" \
    --enforce-eager --max-num-seqs 64 \
    --gpu-memory-utilization 0.90 --dtype half --max-model-len 4096 \
    2>&1 | tee "$LOG_DIR/vllm_8000.log"
