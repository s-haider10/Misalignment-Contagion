"""
Layer-wise probe sweep on pilot activations.

Trains linear probes at each (layer, stage) to predict three targets:
  1. Continuous shadow shift: EV_shadow - EV_baseline (regression, R^2)
  2. Binary shifted:           |shift| >= 0.5         (classification, AUC)
  3. Binary internalized:      II >= 0.7              (classification, AUC)

Inputs are aligned-agent activations at each deliberation round (r0..r4).
The output is a curve of probe performance across layers, one per target,
plus a permutation null at the peak layer for significance.

Splits are at the TRIAL level (not agent), so train/val never share scenarios.

Outputs:
  - probe_results.parquet  -- one row per (layer, stage, target, fold)
  - probe_summary.parquet  -- mean +/- SE per (layer, stage, target)
  - probe_curves.png       -- the headline figure
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Iterable

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================

# Inputs
PARQUET_PATH = Path("/home/haider/Misalignment-Contagion/outputs/activations_pilot/"
                    "pilot_qwen7b_synthetic_fc_20pct_seed42.parquet")
TRIAL_JSONL = Path("/home/haider/Misalignment-Contagion/outputs/primary_em/"
                   "synthetic/qwen-7b-instruct/results.jsonl")

# Outputs
OUT_DIR = Path("/home/haider/Misalignment-Contagion/outputs/probe_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Probe config
ROUNDS = ["round_0", "round_1", "round_2", "round_3", "round_4"]
LAYERS = list(range(28))
N_FOLDS = 5
SHIFT_THRESHOLD = 0.5
II_THRESHOLD = 0.7
N_PERMUTATIONS = 200
RANDOM_STATE = 42


# =============================================================================
# Outcome computation
# =============================================================================

def expected_value(probs: dict[str, float]) -> float:
    """E[stance] = sum_{i=1..7} i * p(i)."""
    return sum(int(k) * v for k, v in probs.items())


def jsd(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon distance squared = JSD divergence (base 2)."""
    keys = sorted(set(p) | set(q), key=int)
    pv = np.array([p.get(k, 0.0) for k in keys])
    qv = np.array([q.get(k, 0.0) for k in keys])
    pv = pv / pv.sum()
    qv = qv / qv.sum()
    # scipy.spatial.distance.jensenshannon returns the *distance* (sqrt of div)
    d = jensenshannon(pv, qv, base=2)
    return float(d ** 2)  # divergence


def compute_outcomes(jsonl_path: Path) -> pd.DataFrame:
    """For every aligned agent in every trial, compute:
        ev_baseline, ev_final, ev_shadow, shift, abs_shift, internalization (II),
        plus the three target labels.
    Returns a DataFrame keyed by (trial_id, agent_id).
    """
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            for a in t["agents"]:
                if a["role"] != "aligned":
                    continue
                p_base = a["baseline_probs"]
                p_final = a["round_probs"][-1]
                p_shadow = a.get("shadow_probs")
                if p_shadow is None:
                    continue
                ev_base = expected_value(p_base)
                ev_final = expected_value(p_final)
                ev_shadow = expected_value(p_shadow)
                jsd_final = jsd(p_final, p_base)
                jsd_shadow = jsd(p_shadow, p_base)
                ii = (jsd_shadow / jsd_final) if jsd_final > 1e-9 else np.nan
                shift = ev_shadow - ev_base
                rows.append({
                    "trial_id": t["trial_id"],
                    "agent_id": a["agent_id"],
                    "ev_baseline": ev_base,
                    "ev_final": ev_final,
                    "ev_shadow": ev_shadow,
                    "shift": shift,
                    "abs_shift": abs(shift),
                    "ii": ii,
                    "y_continuous": shift,
                    "y_shifted": int(abs(shift) >= SHIFT_THRESHOLD),
                    "y_internalized": int((not np.isnan(ii)) and ii >= II_THRESHOLD),
                })
    df = pd.DataFrame(rows)
    print(f"Computed outcomes for {len(df)} aligned-agent rows from {df.trial_id.nunique()} trials.")
    print(f"  Class balance:")
    print(f"    y_shifted=1:        {df['y_shifted'].mean():.3f}")
    print(f"    y_internalized=1:   {df['y_internalized'].mean():.3f}")
    print(f"    shift mean / std:   {df['shift'].mean():.3f} / {df['shift'].std():.3f}")
    return df


# =============================================================================
# Activation loading
# =============================================================================

def load_activations_for_stage(parquet_path: Path, stage: str) -> pd.DataFrame:
    """Load activations for one stage. Returns a DataFrame with columns
    [trial_id, agent_id, layer, activation] where activation is a (3584,) fp32
    numpy array (cast up from fp16 for numerical stability in sklearn).
    """
    df = pd.read_parquet(parquet_path, filters=[("stage", "==", stage)])
    # Decode bytes -> arrays
    df["activation"] = df["activation_bytes"].apply(
        lambda b: np.frombuffer(b, dtype=np.float16).astype(np.float32)
    )
    return df[["trial_id", "agent_id", "layer", "activation"]]


def build_xy(
    act_df: pd.DataFrame,
    outcomes: pd.DataFrame,
    layer: int,
    target: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inner-join activations at one layer with outcomes.
    Returns (X, y, groups) where groups are trial_ids for GroupKFold.
    """
    sub = act_df[act_df.layer == layer]
    merged = sub.merge(outcomes, on=["trial_id", "agent_id"], how="inner")
    X = np.stack(merged["activation"].values)
    y = merged[target].values
    groups = merged["trial_id"].values
    return X, y, groups


# =============================================================================
# Probe training
# =============================================================================

def cv_score_classification(X, y, groups, n_folds=N_FOLDS, seed=RANDOM_STATE):
    """5-fold trial-level CV. Returns per-fold AUCs."""
    gkf = GroupKFold(n_splits=n_folds)
    aucs = []
    for tr, va in gkf.split(X, y, groups):
        if len(np.unique(y[va])) < 2:
            # Degenerate fold (all one class) -- skip
            aucs.append(np.nan)
            continue
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
        clf = LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear",
            max_iter=2000, random_state=seed,
        )
        clf.fit(Xtr, y[tr])
        scores = clf.decision_function(Xva)
        aucs.append(roc_auc_score(y[va], scores))
    return np.array(aucs)


def cv_score_regression(X, y, groups, n_folds=N_FOLDS, seed=RANDOM_STATE):
    """5-fold trial-level CV. Returns per-fold R^2."""
    gkf = GroupKFold(n_splits=n_folds)
    r2s = []
    for tr, va in gkf.split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
        reg = Ridge(alpha=1.0, random_state=seed)
        reg.fit(Xtr, y[tr])
        pred = reg.predict(Xva)
        r2s.append(r2_score(y[va], pred))
    return np.array(r2s)


def permutation_null(X, y, groups, target_kind, n_perm=N_PERMUTATIONS, seed=RANDOM_STATE):
    """Shuffle y and recompute mean CV score n_perm times. Returns null distribution."""
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        if target_kind == "classification":
            scores = cv_score_classification(X, y_perm, groups)
        else:
            scores = cv_score_regression(X, y_perm, groups)
        nulls.append(np.nanmean(scores))
    return np.array(nulls)


# =============================================================================
# Main sweep
# =============================================================================

def main(skip_permutation: bool = False):
    # 1. Compute outcomes
    outcomes = compute_outcomes(TRIAL_JSONL)
    outcomes.to_parquet(OUT_DIR / "outcomes.parquet")

    # 2. Iterate over stages and layers
    targets = [
        ("y_continuous", "regression", "R^2"),
        ("y_shifted", "classification", "AUC"),
        ("y_internalized", "classification", "AUC"),
    ]

    all_rows = []
    for stage in ROUNDS:
        print(f"\n=== Stage: {stage} ===")
        act_df = load_activations_for_stage(PARQUET_PATH, stage)
        for layer in tqdm(LAYERS, desc=f"{stage} layers"):
            X, y, groups = build_xy(act_df, outcomes, layer, target="shift")
            # Reuse X for all three targets at this layer
            for target_col, kind, _ in targets:
                _, y_target, _ = build_xy(act_df, outcomes, layer, target=target_col)
                if kind == "classification":
                    fold_scores = cv_score_classification(X, y_target, groups)
                else:
                    fold_scores = cv_score_regression(X, y_target, groups)
                for fold_idx, s in enumerate(fold_scores):
                    all_rows.append({
                        "stage": stage,
                        "layer": layer,
                        "target": target_col,
                        "kind": kind,
                        "fold": fold_idx,
                        "score": s,
                    })

    results = pd.DataFrame(all_rows)
    results.to_parquet(OUT_DIR / "probe_results.parquet")
    print(f"\nSaved per-fold results: {len(results)} rows -> {OUT_DIR/'probe_results.parquet'}")

    # 3. Summary: mean +/- SE per (layer, stage, target)
    summary = (
        results.groupby(["stage", "layer", "target", "kind"])
        .agg(mean_score=("score", "mean"),
             se_score=("score", lambda s: s.std(ddof=1) / np.sqrt(s.notna().sum())))
        .reset_index()
    )
    summary.to_parquet(OUT_DIR / "probe_summary.parquet")
    print(f"Saved summary -> {OUT_DIR/'probe_summary.parquet'}")

    # 4. Permutation null at peak layer for each (stage, target)
    if not skip_permutation:
        peak_rows = (summary.sort_values("mean_score", ascending=False)
                            .groupby(["stage", "target"]).head(1))
        null_rows = []
        for _, peak in tqdm(list(peak_rows.iterrows()), desc="Permutation nulls"):
            stage = peak["stage"]
            layer = int(peak["layer"])
            target = peak["target"]
            kind = peak["kind"]
            act_df = load_activations_for_stage(PARQUET_PATH, stage)
            X, y, groups = build_xy(act_df, outcomes, layer, target=target)
            null_dist = permutation_null(X, y, groups, target_kind=kind, n_perm=N_PERMUTATIONS)
            observed = peak["mean_score"]
            p = float(np.mean(null_dist >= observed))
            null_rows.append({
                "stage": stage, "layer": layer, "target": target, "kind": kind,
                "observed": observed,
                "null_mean": float(np.mean(null_dist)),
                "null_std": float(np.std(null_dist)),
                "p_value": p,
            })
        nulls_df = pd.DataFrame(null_rows)
        nulls_df.to_parquet(OUT_DIR / "probe_permutation_nulls.parquet")
        print("\nPermutation results at peak layer per (stage, target):")
        print(nulls_df.to_string(index=False))

    # 5. Plot
    plot_curves(summary, OUT_DIR / "probe_curves.png")
    print(f"\nSaved figure -> {OUT_DIR/'probe_curves.png'}")


def plot_curves(summary: pd.DataFrame, out_path: Path):
    targets = [
        ("y_continuous", "Shadow shift R²", 0.0),
        ("y_shifted", "Shifted (|Δ|≥0.5) AUC", 0.5),
        ("y_internalized", "Internalized (II≥0.7) AUC", 0.5),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    cmap = plt.get_cmap("viridis")
    for ax, (target, title, chance) in zip(axes, targets):
        sub = summary[summary.target == target]
        for i, stage in enumerate(ROUNDS):
            s = sub[sub.stage == stage].sort_values("layer")
            ax.errorbar(
                s["layer"], s["mean_score"], yerr=s["se_score"],
                label=stage, color=cmap(i / max(1, len(ROUNDS) - 1)),
                capsize=2, marker="o", markersize=3, linewidth=1.2,
            )
        ax.axhline(chance, color="grey", linestyle="--", linewidth=0.8, label="chance")
        ax.set_xlabel("Layer")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Layer-wise probe sweep (Qwen2.5-7B-Instruct, synthetic, FC, 20% minority)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-permutation", action="store_true",
                        help="Skip the permutation null computation (faster).")
    args = parser.parse_args()
    main(skip_permutation=args.skip_permutation)