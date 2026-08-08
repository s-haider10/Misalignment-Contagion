"""Pre-create claim files for trial_ids already present in the master file.

graph_run uses a claim file per trial_id to dedupe between workers. If the
claim file exists AND the trial is in any shard file, the worker skips it
in O(1). By pre-populating claims for trials already in the master, we
guarantee fresh workers skip the done set immediately at startup.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import time
from pathlib import Path

MASTER = Path("outputs/master")
RUNS = Path("outputs/graph_features/runs")


def safe_name(trial_id: str) -> str:
    """Match graph_run.try_claim's sanitization."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", trial_id)


def pre_claim(dataset: str, wipe_existing_claims: bool = True) -> dict:
    master_file = MASTER / f"{dataset}.jsonl"
    if not master_file.exists():
        return {"dataset": dataset, "status": "no_master", "claimed": 0}

    claims_dir = RUNS / dataset / "claims"
    if wipe_existing_claims and claims_dir.is_dir():
        # Wipe stale claims; we'll repopulate ones for done trials
        for p in claims_dir.iterdir():
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    claims_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(master_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = rec.get("trial_id")
            if not tid:
                continue
            claim_path = claims_dir / f"{safe_name(tid)}.claim"
            try:
                fd = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                os.write(fd, b"pre-claimed (master)")
                os.close(fd)
                n += 1
            except FileExistsError:
                n += 1  # already claimed, count it
    return {"dataset": dataset, "status": "ok", "claimed": n,
            "claims_dir": str(claims_dir)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--keep-claims", action="store_true",
                        help="Don't wipe existing claim files before re-creating.")
    args = parser.parse_args()
    for ds in args.datasets:
        res = pre_claim(ds, wipe_existing_claims=not args.keep_claims)
        if res["status"] != "ok":
            print(f"[{ds}] {res['status']}")
            continue
        print(f"[{ds}] pre-claimed {res['claimed']:,} done trials -> {res['claims_dir']}")


if __name__ == "__main__":
    main()
