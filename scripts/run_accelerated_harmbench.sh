#!/bin/bash
# Accelerated harmbench finisher (v2: deterministic sharding instead of claims).
#
# 8 workers, each handling 1/8 of the trial queue via --shard-id i --num-shards 8.
# Pre-existing master file trials are still skipped via the claim system
# (pre_claim_done.py pre-fills claim files for the done trial_ids).
#
# Existing accelerated vLLMs must be healthy on ports 8000/8010/8020/8030.
set -uo pipefail
cd /home/haider/Misalignment-Contagion
export PYTHONPATH=.
PY=/home/haider/Misalignment-Contagion/.venv/bin/python

LOG_DIR=logs/accel
mkdir -p "$LOG_DIR"

DATASETS=(harmbench_contextual harmbench_copyright)
N_SHARDS=8

run_dataset() {
    local ds=$1
    echo "[$(date)] === ${ds}: pre-claiming ${ds} done trials from master ==="
    "$PY" scripts/pre_claim_done.py "$ds"

    echo "[$(date)] === ${ds}: launching ${N_SHARDS} parallel shards ==="
    declare -a pids=()
    for shard_id in $(seq 0 $((N_SHARDS - 1))); do
        # Round-robin GPU assignment
        local gpu=$((shard_id / 2))
        local port=$((8000 + gpu * 10))
        local shard_tag="accel_shard${shard_id}of${N_SHARDS}"
        local logf="${LOG_DIR}/${ds}_${shard_tag}.log"
        MC_MAX_TOKENS=256 nohup "$PY" -m misalignment_contagion.graph_run \
            --dataset "$ds" \
            --n-scenarios 40 \
            --seed 42 \
            --model-key qwen-7b-instruct \
            --model-condition model_induced \
            --base-port "$port" \
            --concurrency 24 \
            --shard-tag "$shard_tag" \
            --shard-id "$shard_id" \
            --num-shards "$N_SHARDS" \
            >> "$logf" 2>&1 &
        pids+=($!)
        echo "  shard ${shard_id}/${N_SHARDS} (gpu $gpu port $port) PID $!"
    done

    echo "[$(date)] ${ds}: ${N_SHARDS} workers running, waiting for completion..."
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=$((failed+1))
            echo "  PID $pid exited non-zero"
        fi
    done
    echo "[$(date)] ${ds}: done ($failed failed shards)"

    echo "[$(date)] ${ds}: consolidating master file"
    "$PY" scripts/consolidate_master.py "$ds"
}

echo "[$(date)] ACCELERATED HARMBENCH START"
for ds in "${DATASETS[@]}"; do
    run_dataset "$ds"
done
echo "[$(date)] ACCELERATED HARMBENCH DONE"
