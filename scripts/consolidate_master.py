"""Consolidate per-shard graph_run output into master JSONL per dataset.

Reads outputs/graph_features/runs/<dataset>/results.gpu*.jsonl shard files,
deduplicates by trial_id (preferring shard with the longer record / latest
timestamp on tie), and writes outputs/master/<dataset>.jsonl.

Robustness sweep shards (results.robust_*) are intentionally NOT included.

Usage:
    python scripts/consolidate_master.py [datasets...]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict

RUNS = Path("outputs/graph_features/runs")
OUT = Path("outputs/master")
DATASETS_DEFAULT = ["synthetic", "moral_stories", "harmbench_standard",
                    "harmbench_contextual", "harmbench_copyright"]


def consolidate(ds: str) -> dict:
    in_dir = RUNS / ds
    if not in_dir.is_dir():
        return {"dataset": ds, "status": "no_dir", "unique": 0, "total_lines": 0}
    # Include all results.*.jsonl shard files EXCEPT robustness sweep shards
    # (those have a different model_key/seed and live in their own analyses).
    shards = sorted(
        p for p in in_dir.glob("results.*.jsonl")
        if not p.name.startswith("results.robust_")
    )
    if not shards:
        return {"dataset": ds, "status": "no_shards", "unique": 0, "total_lines": 0}

    best: dict[str, tuple[int, dict, str]] = {}  # trial_id -> (len, record, source_shard)
    total_lines = 0
    for shard in shards:
        with open(shard) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = rec.get("trial_id")
                if not tid:
                    continue
                rec_len = len(line)
                # Keep the longest (most complete) version
                if tid not in best or rec_len > best[tid][0]:
                    best[tid] = (rec_len, rec, shard.name)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{ds}.jsonl"
    with open(out_path, "w") as f:
        for tid in sorted(best.keys()):
            f.write(json.dumps(best[tid][1]) + "\n")

    src_breakdown = defaultdict(int)
    for _, _, src in best.values():
        src_breakdown[src] += 1

    return {
        "dataset": ds,
        "status": "ok",
        "unique": len(best),
        "total_lines": total_lines,
        "duplicate_frac": (total_lines - len(best)) / max(total_lines, 1),
        "out": str(out_path),
        "shards": dict(src_breakdown),
    }


def main():
    datasets = sys.argv[1:] if len(sys.argv) > 1 else DATASETS_DEFAULT
    for ds in datasets:
        res = consolidate(ds)
        if res["status"] != "ok":
            print(f"[{ds}] {res['status']}")
            continue
        print(f"[{ds}] unique={res['unique']:,}  raw_lines={res['total_lines']:,}  "
              f"dup={res['duplicate_frac']:.1%}  -> {res['out']}")
        for src, n in sorted(res["shards"].items()):
            print(f"    {src:>40}: {n:,}")


if __name__ == "__main__":
    main()
