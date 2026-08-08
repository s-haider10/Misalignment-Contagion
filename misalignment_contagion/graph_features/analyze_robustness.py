"""Analyze robustness sweep results.

Aggregates the per-(model, seed) graph_run outputs from the 20-graph subset,
joins with manifest features, runs 5-fold CV with Strategy C (topology +
dataset dummies) and Strategy D (baseline: dataset + minority_ratio only),
and reports R^2 with variance bars across:
  * Qwen-7B seeds 42, 123, 456    (in-model variance)
  * Llama-8B seed 42               (cross-family robustness)
  * Qwen-14B seed 42               (scale robustness)

Outputs:
  outputs/robustness/r2_comparison.csv
  outputs/robustness/r2_comparison.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from misalignment_contagion.graph_features.extract import aligned_shadow_shifts


RUNS = Path("outputs/graph_features/runs")
MANIFEST = Path("outputs/graph_features/graph_manifest_subset/subset_features.csv")
DATASETS = ["synthetic", "moral_stories"]


def collect_one(model_key: str, seed: int) -> pd.DataFrame:
    """Build a per-(graph_id, dataset) outcome dataframe for one (model, seed)."""
    by_gd = defaultdict(list)
    n_trials = defaultdict(int)
    for ds in DATASETS:
        ds_dir = RUNS / ds
        if not ds_dir.is_dir():
            continue
        # Shard tag pattern: robust_<phase>_<model>_s<seed>
        for shard in sorted(ds_dir.glob(f"results.robust_*_{model_key}_s{seed}.jsonl")):
            with open(shard) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        trial = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if trial.get("model_key") != model_key:
                        continue
                    shifts = aligned_shadow_shifts(trial)
                    if not shifts:
                        continue
                    key = (trial["graph_id"], ds)
                    by_gd[key].extend(shifts)
                    n_trials[key] += 1

    rows = []
    for (gid, ds), shifts_list in by_gd.items():
        shifts = np.array(shifts_list)
        rows.append({
            "graph_id": gid,
            "dataset": ds,
            "n_trials": n_trials[(gid, ds)],
            "n_aligned_obs": len(shifts),
            "y_mean_shadow_shift": float(shifts.mean()),
            "y_var_shadow_shift": float(shifts.var(ddof=1)) if len(shifts) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def join_features(df: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST)
    return df.merge(manifest, on="graph_id", how="left", validate="many_to_one")


def fit_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42) -> float:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yt_all, yp_all = [], []
    for tr, te in kf.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        m = LassoCV(alphas=np.logspace(-3, 1, 50), cv=3, max_iter=20000, random_state=42)
        m.fit(Xtr, y[tr])
        yp_all.append(m.predict(Xte))
        yt_all.append(y[te])
    return r2_score(np.concatenate(yt_all), np.concatenate(yp_all))


def compute_r2s(df: pd.DataFrame) -> dict:
    """Return dict with R^2 for Strategy C (topology + dataset) and D (baseline)."""
    if df.empty or len(df) < 10:
        return {"r2_C_mean": np.nan, "r2_D_mean": np.nan,
                "r2_C_var": np.nan, "r2_D_var": np.nan, "n_rows": len(df)}

    drop = {"graph_id", "family", "dataset", "placement_summary",
            "n_trials", "n_aligned_obs",
            "y_mean_shadow_shift", "y_var_shadow_shift", "y_sem_shadow_shift"}
    topo_cols = [c for c in df.columns if c not in drop
                 and np.issubdtype(df[c].dtype, np.number)
                 and c != "minority_ratio"]

    dummies = pd.get_dummies(df["dataset"], prefix="dataset",
                              drop_first=True).astype(float).values
    X_baseline = np.hstack([df[["minority_ratio"]].values, dummies])
    X_full = np.hstack([X_baseline, df[topo_cols].values])

    out = {"n_rows": len(df)}
    for target, suffix in [("y_mean_shadow_shift", "mean"),
                            ("y_var_shadow_shift", "var")]:
        y = df[target].values
        out[f"r2_C_{suffix}"] = fit_cv(X_full, y)
        out[f"r2_D_{suffix}"] = fit_cv(X_baseline, y)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/robustness"))
    parser.add_argument("--include-original-qwen7b-seed42", action="store_true",
                        help="Pull seed=42 Qwen-7B data from the main run for comparison.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        ("qwen-7b-instruct", 42),
        ("qwen-7b-instruct", 123),
        ("qwen-7b-instruct", 456),
        ("llama-8b-instruct", 42),
        ("qwen-14b-instruct", 42),
    ]

    rows = []
    for model_key, seed in runs:
        print(f"=== {model_key} seed={seed} ===")
        df = collect_one(model_key, seed)
        if df.empty and (model_key, seed) == ("qwen-7b-instruct", 42):
            # The original gpu0..3 run used seed=42 with non-robust shard tags
            # Fall back to all qwen-7b shards in synthetic + moral_stories
            print("  (no robust shards; pulling from main run with seed=42)")
            for ds in DATASETS:
                ds_dir = RUNS / ds
                if not ds_dir.is_dir():
                    continue
                for shard in sorted(ds_dir.glob("results.gpu*.jsonl")):
                    with open(shard) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                trial = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if trial.get("model_key") != model_key:
                                continue
                            if trial.get("seed") != seed:
                                continue
                            shifts = aligned_shadow_shifts(trial)
                            if not shifts:
                                continue
            # Re-collect for the original run path
            df = pd.read_csv(
                "outputs/graph_features/joined_no_harmbench/per_graph_per_dataset.csv"
            )
            # Restrict to the 20-graph robustness subset for apples-to-apples
            subset_ids = set(pd.read_csv(
                "outputs/graph_features/robustness_subset/robustness_graphs.csv"
            )["graph_id"])
            df = df[df["graph_id"].isin(subset_ids)].copy()
        else:
            df = join_features(df)

        print(f"  rows={len(df)}  trials={df['n_trials'].sum() if 'n_trials' in df.columns else 0}")
        if df.empty:
            print("  (no data, skipping)")
            continue
        r2s = compute_r2s(df)
        r2s["model_key"] = model_key
        r2s["seed"] = seed
        rows.append(r2s)
        print(f"  R2_C_mean={r2s['r2_C_mean']:.3f}  R2_D_mean={r2s['r2_D_mean']:.3f}  "
              f"Δmean={r2s['r2_C_mean']-r2s['r2_D_mean']:+.3f}")
        print(f"  R2_C_var={r2s['r2_C_var']:.3f}   R2_D_var={r2s['r2_D_var']:.3f}   "
              f"Δvar={r2s['r2_C_var']-r2s['r2_D_var']:+.3f}")

    if not rows:
        print("No data to analyze.")
        return

    out_df = pd.DataFrame(rows)[["model_key", "seed", "n_rows",
                                   "r2_C_mean", "r2_D_mean",
                                   "r2_C_var", "r2_D_var"]]
    out_df["delta_mean"] = out_df["r2_C_mean"] - out_df["r2_D_mean"]
    out_df["delta_var"] = out_df["r2_C_var"] - out_df["r2_D_var"]
    out_csv = args.out_dir / "r2_comparison.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")
    print(out_df.to_string(index=False))

    # Plot: bar chart with delta R^2 per (model, seed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [f"{r['model_key']}\nseed={r['seed']}" for _, r in out_df.iterrows()]
    x = np.arange(len(labels))
    width = 0.35

    for ax, target in zip(axes, ["mean", "var"]):
        ax.bar(x - width/2, out_df[f"r2_D_{target}"],
               width, label="Baseline (dataset + ratio)", color="steelblue")
        ax.bar(x + width/2, out_df[f"r2_C_{target}"],
               width, label="+ Topology features", color="indianred")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("CV R²")
        ax.set_title(f"y_{target}_shadow_shift")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylim(min(0, out_df[[f"r2_C_{target}", f"r2_D_{target}"]].min().min() - 0.05), 1.0)

    fig.suptitle("Robustness: does topology add over baseline?", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out_dir / "r2_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out_dir / 'r2_comparison.png'}")


if __name__ == "__main__":
    main()
