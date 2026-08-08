#!/bin/bash
# Phase 2 orchestrator: Qwen-7B seed 456 on syn+moral only.
# Synthetic shards are already running (PIDs in /tmp/p2_syn_pids).
# This script waits for them, then launches moral_stories.
set -uo pipefail
cd /home/haider/Misalignment-Contagion
export PYTHONPATH=.
PY=/home/haider/Misalignment-Contagion/.venv/bin/python

LOG_DIR=logs/robust_ext_p2
mkdir -p "$LOG_DIR"
MANIFEST=outputs/graph_features/robustness_subset/remaining_131_manifest.json

echo "[$(date)] PHASE 2 ORCHESTRATOR START"

# Wait for the synthetic shards launched separately
SYN_PIDS=(340585 340587)
echo "[$(date)] waiting for synthetic shards: ${SYN_PIDS[*]}"
for pid in "${SYN_PIDS[@]}"; do
    wait "$pid" 2>/dev/null
    echo "  synthetic PID $pid exited $?"
done

echo "[$(date)] === synthetic done; launching moral_stories ==="
MS_PIDS=()
for shard in 0 1; do
    port=$([ "$shard" -eq 0 ] && echo 8000 || echo 8010)
    logf="${LOG_DIR}/qwen_s456_moral_stories_shard${shard}of2.log"
    MC_MAX_TOKENS=256 nohup "$PY" -m misalignment_contagion.graph_run \
        --dataset moral_stories \
        --manifest "$MANIFEST" \
        --n-scenarios 10 \
        --seed 456 \
        --model-key qwen-7b-instruct \
        --model-condition model_induced \
        --base-port "$port" \
        --concurrency 24 \
        --shard-tag "robust_ext_qwen-7b-instruct_s456_s${shard}" \
        --shard-id "$shard" \
        --num-shards 2 \
        >> "$logf" 2>&1 &
    MS_PIDS+=($!)
    echo "  moral_stories shard $shard PID $! on port $port"
done

for pid in "${MS_PIDS[@]}"; do
    wait "$pid" 2>/dev/null
    echo "  moral_stories PID $pid exited $?"
done

echo "[$(date)] PHASE 2 ORCHESTRATOR DONE"
