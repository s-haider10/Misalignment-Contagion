#!/bin/bash
# Watches a goutham PID; when it exits, immediately launches a vLLM +
# graph_run worker on the freed GPU. Use one instance per (PID, GPU) pair.
#
# Usage: grab_gpu_and_run.sh <watch_pid> <cuda_device>
set -euo pipefail
cd /home/haider/Misalignment-Contagion

WATCH_PID="$1"
GPU="$2"
PORT=$((8000 + GPU * 10))
TAG="gpu${GPU}"
LOG="/tmp/gpu_reserve_logs/grab_gpu${GPU}_after_${WATCH_PID}.log"
mkdir -p /tmp/gpu_reserve_logs

echo "[$(date)] watching PID $WATCH_PID -> will launch graph_run worker on GPU $GPU (port $PORT) when it exits" | tee -a "$LOG"

while [ -e "/proc/$WATCH_PID" ]; do
    sleep 30
done

echo "[$(date)] PID $WATCH_PID exited; launching graph_run worker on GPU $GPU" | tee -a "$LOG"

# Hand off to the launcher (it stays attached so this script's tmux pane
# shows live progress).
exec bash /home/haider/Misalignment-Contagion/scripts/launch_graph_run_worker.sh \
    "$GPU" "$PORT" "$TAG"
