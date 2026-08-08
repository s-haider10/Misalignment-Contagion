#!/bin/bash
# Extended robustness sweep on the 131 graphs not in the 20-graph subset.
#
# 4 vLLMs split: GPUs 0,1 = Qwen-7B, GPUs 2,3 = Llama-3.1-8B.
# Each (model, seed) combo gets 2 deterministic shards across its 2 vLLMs.
#
# Combos:
#   Qwen-7B seed 123  (2 shards on ports 8000, 8010)
#   Qwen-7B seed 456  (2 shards on ports 8000, 8010)
#   Llama-8B seed 42  (2 shards on ports 8020, 8030)
#
# Total parallel: 6 shards at any one time (2 per active model).
# We run Qwen-7B-s123 + Llama-8B simultaneously, then Qwen-7B-s456 + Llama-8B
# already done → just Qwen-7B-s456.
set -uo pipefail
cd /home/haider/Misalignment-Contagion
export PYTHONPATH=.
PY=/home/haider/Misalignment-Contagion/.venv/bin/python

MANIFEST=outputs/graph_features/robustness_subset/remaining_131_manifest.json
LOG_DIR=logs/robust_ext
mkdir -p "$LOG_DIR"

launch_shards() {
    local model_key=$1
    local seed=$2
    local port_a=$3
    local port_b=$4
    local tag="robust_ext_${model_key}_s${seed}"
    echo "[$(date)] launching 2 shards for ${tag} on ports ${port_a}, ${port_b}"
    declare -ga LAUNCHED_PIDS=()
    for shard_id in 0 1; do
        local port=$([ "$shard_id" -eq 0 ] && echo "$port_a" || echo "$port_b")
        local logf="${LOG_DIR}/${tag}_shard${shard_id}of2.log"
        MC_MAX_TOKENS=256 nohup "$PY" -m misalignment_contagion.graph_run \
            --dataset all \
            --manifest "$MANIFEST" \
            --n-scenarios 10 \
            --seed "$seed" \
            --model-key "$model_key" \
            --model-condition model_induced \
            --base-port "$port" \
            --concurrency 24 \
            --shard-tag "${tag}_s${shard_id}" \
            --shard-id "$shard_id" \
            --num-shards 2 \
            >> "$logf" 2>&1 &
        LAUNCHED_PIDS+=($!)
        echo "  shard ${shard_id}/2 (port ${port}) PID $!"
    done
}

wait_all() {
    local -n pids=$1
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=$((failed + 1))
            echo "  PID $pid exited non-zero"
        fi
    done
    echo "  (${failed} failed shards)"
}

echo "[$(date)] EXTENDED ROBUSTNESS SWEEP START"

# Run all 3 combos at once: Qwen-7B-s123 (GPUs 0,1), Qwen-7B-s456 will reuse
# the same vLLMs after, Llama-8B-s42 (GPUs 2,3) runs in parallel.

# Phase 1: Qwen-7B-s123 (GPUs 0,1) + Llama-8B-s42 (GPUs 2,3) in parallel
echo "[$(date)] === Phase 1: Qwen-7B-s123 + Llama-8B-s42 ==="
launch_shards qwen-7b-instruct  123 8000 8010
QWEN123_PIDS=("${LAUNCHED_PIDS[@]}")
launch_shards llama-8b-instruct  42 8020 8030
LLAMA_PIDS=("${LAUNCHED_PIDS[@]}")

echo "[$(date)] waiting for all 4 shards..."
ALL_P1=("${QWEN123_PIDS[@]}" "${LLAMA_PIDS[@]}")
wait_all ALL_P1

# Phase 2: Qwen-7B-s456 (GPUs 0,1 freed)
echo "[$(date)] === Phase 2: Qwen-7B-s456 ==="
launch_shards qwen-7b-instruct 456 8000 8010
QWEN456_PIDS=("${LAUNCHED_PIDS[@]}")
wait_all QWEN456_PIDS

echo "[$(date)] EXTENDED ROBUSTNESS SWEEP DONE"
