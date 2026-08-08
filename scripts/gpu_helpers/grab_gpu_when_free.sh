#!/bin/bash
# Watches a goutham PID; when it exits, launches a GPU reservation on the
# specified CUDA device. Use one instance per (PID, GPU) pair.
#
# Usage: grab_gpu_when_free.sh <watch_pid> <cuda_device>
set -euo pipefail
cd /home/haider/Misalignment-Contagion

WATCH_PID="$1"
GPU="$2"
LOG="/tmp/gpu_reserve_logs/grab_gpu${GPU}_after_${WATCH_PID}.log"
mkdir -p /tmp/gpu_reserve_logs

echo "[$(date)] watching PID $WATCH_PID -> will reserve GPU $GPU when it exits" | tee -a "$LOG"

while [ -e "/proc/$WATCH_PID" ]; do
    sleep 30
done

echo "[$(date)] PID $WATCH_PID exited; launching reservation on GPU $GPU" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="$GPU" nohup \
    /home/haider/Misalignment-Contagion/.venv/bin/python \
    /home/haider/Misalignment-Contagion/scripts/gpu_helpers/reserve_gpu.py 0 \
    > "/tmp/gpu_reserve_logs/gpu${GPU}.log" 2>&1 &
RESERVE_PID=$!
echo "[$(date)] reservation PID $RESERVE_PID on GPU $GPU" | tee -a "$LOG"
