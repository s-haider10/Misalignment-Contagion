#!/bin/bash
# Watcher: polls shadow_self_hidden_ablation results.jsonl files; once a
# dataset's file has been stable (unchanged size) for STABLE_SECS, runs
# the comparison/plot script for that dataset and then never touches it
# again. Exits once all 5 datasets are processed.
set -euo pipefail
cd /home/haider/Misalignment-Contagion
export PYTHONPATH=.
PY=/home/haider/Misalignment-Contagion/.venv/bin/python

DATASETS=(synthetic moral_stories harmbench_standard harmbench_contextual harmbench_copyright)
MODEL_KEY="qwen-7b-instruct"
ROOT="outputs/shadow_self_hidden_ablation"
STABLE_SECS=180     # file unchanged for this many seconds = "done"
POLL_SECS=30
DONE_FLAG_DIR="/tmp/shadow_self_hidden_plots_done"
mkdir -p "$DONE_FLAG_DIR"

declare -A last_size
declare -A last_change_t

while true; do
    pending=0
    for ds in "${DATASETS[@]}"; do
        flag="$DONE_FLAG_DIR/${ds}.done"
        [[ -f "$flag" ]] && continue
        pending=$((pending + 1))

        f="$ROOT/$ds/$MODEL_KEY/results.jsonl"
        if [[ ! -f "$f" ]]; then
            continue
        fi
        size=$(stat -c %s "$f")
        now=$(date +%s)
        prev_size="${last_size[$ds]:-}"
        if [[ "$size" != "$prev_size" ]]; then
            last_size[$ds]=$size
            last_change_t[$ds]=$now
            continue
        fi
        # size unchanged — check elapsed
        elapsed=$(( now - ${last_change_t[$ds]:-$now} ))
        if (( elapsed >= STABLE_SECS )); then
            echo "==== $(date) :: $ds stable for ${elapsed}s, running plots ===="
            if $PY ablations/compare_shadow_self_hidden_vs_primary.py --dataset "$ds"; then
                touch "$flag"
                echo "==== $(date) :: $ds plots done ===="
            else
                echo "==== $(date) :: $ds plots FAILED, will retry after next stable window ===="
                last_change_t[$ds]=$now   # reset so we wait again
            fi
        fi
    done
    if (( pending == 0 )); then
        echo "==== $(date) :: all 5 datasets plotted, watcher exiting ===="
        exit 0
    fi
    sleep "$POLL_SECS"
done
