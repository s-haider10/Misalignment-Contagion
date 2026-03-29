#!/usr/bin/env python3
"""Convert raw datasets into the unified JSONL scenario format.

Output format per line:
    {
        "id": str,
        "category": str,
        "context": str,
        "position_a": str,   # aligned (safe/refuse)
        "position_b": str,   # misaligned (unsafe/comply)
        "harm": str
    }

Data sources (download separately):
    - Moral Stories: data/moral_stories/raw/moral_stories_full.jsonl
      (from https://github.com/demelin/moral_stories)
    - HarmBench:    data/harmbench/raw/standard.csv, contextual.csv, copyright.csv
      (from HuggingFace walledai/HarmBench)
    - Synthetic:    data/synthetic/scenarios.json (hand-authored, already in format)

Usage:
    python scripts/prepare_data.py            # convert all available raw files
    python scripts/prepare_data.py --check    # just verify outputs exist
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

POSITION_A = "Refuse: do not provide the requested information or assistance."
POSITION_B = "Comply: provide the requested information or assistance."

# Map of (raw source, output, converter function name)
CONVERSIONS: list[tuple[str, str, str]] = [
    ("moral_stories/raw/moral_stories_full.jsonl", "moral_stories/moral_stories.jsonl", "moral_stories"),
    ("harmbench/raw/standard.csv",   "harmbench/standard.jsonl",   "harmbench_standard"),
    ("harmbench/raw/contextual.csv", "harmbench/contextual.jsonl", "harmbench_contextual"),
    ("harmbench/raw/copyright.csv",  "harmbench/copyright.jsonl",  "harmbench_copyright"),
]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  Wrote {len(rows):,} rows -> {path.relative_to(ROOT)}")


# ── Converters ────────────────────────────────────────────────────────────────

def convert_moral_stories(src: Path, out: Path) -> None:
    rows = []
    with open(src) as f:
        for i, line in enumerate(f, 1):
            obj = json.loads(line)
            context = obj["situation"].rstrip(".") + ". " + obj["intention"].rstrip(".") + "."
            rows.append({
                "id": f"MS-{i:05d}",
                "category": "moral_stories",
                "context": context,
                "position_a": obj["moral_action"],
                "position_b": obj["immoral_action"],
                "harm": obj["immoral_consequence"],
            })
    write_jsonl(rows, out)


def convert_harmbench_standard(src: Path, out: Path) -> None:
    rows = []
    with open(src) as f:
        for i, row in enumerate(csv.DictReader(f), 1):
            rows.append({
                "id": f"HB-STD-{i:03d}",
                "category": row["category"],
                "context": row["prompt"].rstrip(".") + ".",
                "position_a": POSITION_A,
                "position_b": POSITION_B,
                "harm": row["category"],
            })
    write_jsonl(rows, out)


def convert_harmbench_contextual(src: Path, out: Path) -> None:
    rows = []
    with open(src) as f:
        for i, row in enumerate(csv.DictReader(f), 1):
            context = (
                row["prompt"].rstrip(".") + "."
                + "\n\nBackground: " + row["context"].strip()
            )
            rows.append({
                "id": f"HB-CTX-{i:03d}",
                "category": row["category"],
                "context": context,
                "position_a": POSITION_A,
                "position_b": POSITION_B,
                "harm": row["category"],
            })
    write_jsonl(rows, out)


def convert_harmbench_copyright(src: Path, out: Path) -> None:
    rows = []
    with open(src) as f:
        for i, row in enumerate(csv.DictReader(f), 1):
            rows.append({
                "id": f"HB-CPY-{i:03d}",
                "category": "copyright",
                "context": row["prompt"].rstrip(".") + ".",
                "position_a": POSITION_A,
                "position_b": POSITION_B,
                "harm": "copyright",
            })
    write_jsonl(rows, out)


CONVERTERS = {
    "moral_stories": convert_moral_stories,
    "harmbench_standard": convert_harmbench_standard,
    "harmbench_contextual": convert_harmbench_contextual,
    "harmbench_copyright": convert_harmbench_copyright,
}


# ── Main ──────────────────────────────────────────────────────────────────────

def check_outputs() -> bool:
    """Verify all output JSONLs exist."""
    all_ok = True
    for _, out_rel, _ in CONVERSIONS:
        out = DATA / out_rel
        if out.exists():
            with open(out) as f:
                n = sum(1 for line in f if line.strip())
            print(f"  OK  {out.relative_to(ROOT)} ({n:,} rows)")
        else:
            print(f"  MISSING  {out.relative_to(ROOT)}")
            all_ok = False
    # Synthetic is always present
    syn = DATA / "synthetic" / "scenarios.json"
    if syn.exists():
        with open(syn) as f:
            n = len(json.load(f)["scenarios"])
        print(f"  OK  {syn.relative_to(ROOT)} ({n} scenarios)")
    else:
        print(f"  MISSING  {syn.relative_to(ROOT)}")
        all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for experiments")
    parser.add_argument("--check", action="store_true",
                        help="Just verify outputs exist, don't convert.")
    args = parser.parse_args()

    if args.check:
        print("Checking dataset outputs:")
        ok = check_outputs()
        sys.exit(0 if ok else 1)

    converted = 0
    skipped = 0

    for src_rel, out_rel, converter_key in CONVERSIONS:
        src = DATA / src_rel
        out = DATA / out_rel

        if out.exists():
            print(f"  Already exists: {out.relative_to(ROOT)} (skipping)")
            skipped += 1
            continue

        if not src.exists():
            print(f"  Raw source missing: {src.relative_to(ROOT)} (skipping)")
            skipped += 1
            continue

        print(f"Converting {src.relative_to(ROOT)} ...")
        CONVERTERS[converter_key](src, out)
        converted += 1

    print(f"\nDone. Converted: {converted}, Skipped: {skipped}")
    print("\nFinal check:")
    check_outputs()


if __name__ == "__main__":
    main()
