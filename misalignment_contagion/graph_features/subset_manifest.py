"""
Stratified subsampler for the full graph manifest.

The full manifest (11k+ graphs) is too large to run deliberation on. This
script picks a balanced subset stratified by (family, n_agents, n_misaligned,
placement_summary) so that each parameter cell is represented without any
single family or seed dominating.

Strategy:
  - Group all unique graphs by (family, n_agents, n_misaligned, placement_summary)
  - For each cell, take up to `per_cell` graphs (default 2)
  - Within a cell, prefer graphs whose seeds/parameters are spread out
  - Total target controlled via --target_n (overrides per_cell to hit it)

Outputs:
  subset_manifest.json  : subset of graph specs ready to feed deliberation
  subset_features.csv   : feature matrix for the subset
  subset_breakdown.csv  : per-family / per-cell counts in the subset
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def stratified_subset(manifest: list[dict], df_feat: pd.DataFrame,
                      target_n: int = 250, seed: int = 42) -> tuple[list[dict], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    by_id = {g["graph_id"]: g for g in manifest}

    # Group rows by (family, n_agents, n_misaligned, placement_summary)
    cells: dict[tuple, list[str]] = defaultdict(list)
    for _, row in df_feat.iterrows():
        cell = (row["family"], row["n_agents"], row["n_misaligned"],
                row["placement_summary"])
        cells[cell].append(row["graph_id"])
    n_cells = len(cells)
    print(f"Total cells: {n_cells}")

    chosen_ids: list[str] = []
    if n_cells <= target_n:
        # We can afford ≥1 per cell. Spread the remaining budget.
        per_cell = max(1, target_n // n_cells)
        for ids in cells.values():
            take = min(per_cell, len(ids))
            if take == len(ids):
                chosen_ids.extend(ids)
            else:
                idx = rng.choice(len(ids), size=take, replace=False)
                chosen_ids.extend([ids[i] for i in idx])
    else:
        # More cells than budget: pick a subset of cells, proportional per family.
        by_family: dict[str, list[tuple]] = defaultdict(list)
        for cell, ids in cells.items():
            by_family[cell[0]].append((cell, ids))

        total = sum(len(v) for v in by_family.values())
        for fam, fam_cells in by_family.items():
            n_take = max(1, round(target_n * len(fam_cells) / total))
            n_take = min(n_take, len(fam_cells))
            idx = rng.choice(len(fam_cells), size=n_take, replace=False)
            sampled_cells = [fam_cells[i] for i in idx]
            for cell, ids in sampled_cells:
                # One representative graph per chosen cell
                chosen_ids.append(str(rng.choice(ids)))

    chosen_set = set(chosen_ids)
    subset_manifest = [by_id[gid] for gid in chosen_ids if gid in by_id]
    subset_df = df_feat[df_feat["graph_id"].isin(chosen_set)].reset_index(drop=True)
    return subset_manifest, subset_df


def run(manifest_path: Path, features_path: Path, out_dir: Path,
        target_n: int = 250, seed: int = 42):
    with open(manifest_path) as f:
        manifest = json.load(f)
    df_feat = pd.read_csv(features_path)
    print(f"Loaded {len(manifest)} graphs ({df_feat.shape[1]} feature columns)")

    subset_manifest, subset_df = stratified_subset(manifest, df_feat,
                                                   target_n=target_n, seed=seed)
    print(f"Selected {len(subset_manifest)} graphs (target was {target_n})")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "subset_manifest.json", "w") as f:
        json.dump(subset_manifest, f, indent=2)
    subset_df.to_csv(out_dir / "subset_features.csv", index=False)

    breakdown = subset_df.groupby("family").agg(
        n_graphs=("graph_id", "count"),
        n_distinct_n=("n_agents", "nunique"),
        n_distinct_k=("n_misaligned", "nunique"),
        n_distinct_placements=("placement_summary", "nunique"),
    ).reset_index()
    breakdown.to_csv(out_dir / "subset_breakdown.csv", index=False)
    print("\nSubset breakdown:")
    print(breakdown.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("outputs/graph_manifest/graph_manifest.json"))
    parser.add_argument("--features", type=Path,
                        default=Path("outputs/graph_manifest/graph_features.csv"))
    parser.add_argument("--out_dir", type=Path,
                        default=Path("outputs/graph_manifest_subset"))
    parser.add_argument("--target_n", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.manifest, args.features, args.out_dir,
        target_n=args.target_n, seed=args.seed)