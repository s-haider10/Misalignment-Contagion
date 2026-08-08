"""Join graph_run trial outputs with the graph_features manifest.

Reads sharded jsonl results under outputs/graph_features/runs/{dataset}/,
computes per-trial aligned shadow shifts, pools them per graph_id, and
left-joins with the manifest features CSV.

Outputs:
  per_graph_outcomes.csv         one row per graph_id (pooled across datasets)
  per_graph_per_dataset.csv      one row per (graph_id, dataset)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from misalignment_contagion.graph_features.extract import aligned_shadow_shifts


def iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def collect_dataset(run_dir: Path, dataset: str,
                    model_key: str, model_condition: str,
                    by_graph: dict, by_graph_dataset: dict,
                    n_trials_graph: dict, n_trials_graph_dataset: dict):
    ds_dir = run_dir / dataset
    if not ds_dir.is_dir():
        print(f"  [skip] {ds_dir} not found")
        return 0, 0
    shard_files = sorted(p for p in ds_dir.iterdir() if p.suffix == ".jsonl")
    n_total = n_kept = 0
    for shard in shard_files:
        for trial in iter_jsonl(shard):
            n_total += 1
            if trial.get("model_key") != model_key:
                continue
            if trial.get("model_condition") != model_condition:
                continue
            shifts = aligned_shadow_shifts(trial)
            if not shifts:
                continue
            graph_id = trial["graph_id"]
            by_graph[graph_id].extend(shifts)
            by_graph_dataset[(graph_id, dataset)].extend(shifts)
            n_trials_graph[graph_id] += 1
            n_trials_graph_dataset[(graph_id, dataset)] += 1
            n_kept += 1
    print(f"  {dataset}: {n_total} trials, {n_kept} kept (with valid shadow probs)")
    return n_total, n_kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path,
                        default=Path("outputs/graph_features/runs"),
                        help="Directory containing per-dataset shard subdirs.")
    parser.add_argument("--manifest", type=Path,
                        default=Path("outputs/graph_features/graph_manifest_subset/subset_features.csv"),
                        help="Manifest CSV with one row per graph_id.")
    parser.add_argument("--datasets", nargs="+",
                        default=["synthetic", "moral_stories", "harmbench_standard"],
                        help="Datasets to include.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/graph_features/joined"),
                        help="Where to write joined CSVs.")
    parser.add_argument("--model-key", default="qwen-7b-instruct")
    parser.add_argument("--model-condition", default="model_induced")
    args = parser.parse_args()

    print(f"Loading manifest: {args.manifest}")
    manifest = pd.read_csv(args.manifest)
    print(f"  {len(manifest)} graphs in manifest")

    by_graph: dict[str, list[float]] = defaultdict(list)
    by_graph_dataset: dict[tuple, list[float]] = defaultdict(list)
    n_trials_graph: dict[str, int] = defaultdict(int)
    n_trials_graph_dataset: dict[tuple, int] = defaultdict(int)

    print(f"Reading shards from: {args.run_dir}")
    for ds in args.datasets:
        collect_dataset(
            args.run_dir, ds, args.model_key, args.model_condition,
            by_graph, by_graph_dataset, n_trials_graph, n_trials_graph_dataset,
        )

    # Per-graph pooled outcomes
    rows = []
    for gid, shifts_list in by_graph.items():
        shifts = np.array(shifts_list)
        rows.append({
            "graph_id": gid,
            "n_trials": n_trials_graph[gid],
            "n_aligned_obs": len(shifts),
            "y_mean_shadow_shift": float(shifts.mean()),
            "y_var_shadow_shift": float(shifts.var(ddof=1)) if len(shifts) > 1 else 0.0,
            "y_sem_shadow_shift": (float(shifts.std(ddof=1) / np.sqrt(len(shifts)))
                                   if len(shifts) > 1 else 0.0),
        })
    df_outcomes = pd.DataFrame(rows)

    # Per (graph, dataset)
    rows_ds = []
    for (gid, ds), shifts_list in by_graph_dataset.items():
        shifts = np.array(shifts_list)
        rows_ds.append({
            "graph_id": gid,
            "dataset": ds,
            "n_trials": n_trials_graph_dataset[(gid, ds)],
            "n_aligned_obs": len(shifts),
            "y_mean_shadow_shift": float(shifts.mean()),
            "y_var_shadow_shift": float(shifts.var(ddof=1)) if len(shifts) > 1 else 0.0,
            "y_sem_shadow_shift": (float(shifts.std(ddof=1) / np.sqrt(len(shifts)))
                                   if len(shifts) > 1 else 0.0),
        })
    df_outcomes_ds = pd.DataFrame(rows_ds)

    # Join with manifest features (left join: keep only graphs that have outcomes)
    df_joined = df_outcomes.merge(manifest, on="graph_id", how="left")
    n_missing = df_joined["family"].isna().sum()
    if n_missing:
        print(f"  WARN: {n_missing}/{len(df_joined)} graph_ids in outcomes have no manifest row")

    df_joined_ds = df_outcomes_ds.merge(manifest, on="graph_id", how="left")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_main = args.output_dir / "per_graph_outcomes.csv"
    out_ds = args.output_dir / "per_graph_per_dataset.csv"
    df_joined.to_csv(out_main, index=False)
    df_joined_ds.to_csv(out_ds, index=False)

    print(f"\nWrote {len(df_joined)} rows -> {out_main}")
    print(f"Wrote {len(df_joined_ds)} rows -> {out_ds}")
    print(f"\nCoverage: {len(df_outcomes)}/{len(manifest)} manifest graphs have trials")
    print(f"Total aligned-shadow observations: {int(df_outcomes['n_aligned_obs'].sum())}")
    print("\nPer-dataset trial counts:")
    print(df_outcomes_ds.groupby("dataset")[["n_trials", "n_aligned_obs"]].sum().to_string())


if __name__ == "__main__":
    main()
