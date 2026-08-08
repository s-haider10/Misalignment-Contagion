"""
Probe sweep v2 — pilot signal check.

Design:
  - Within-trial contrast: per trial, take the aligned agent with the largest
    |shadow shift| (positive class) and the smallest |shadow shift| (negative
    class). Drop the middle agents.
  - 50 trials -> 50 positives + 50 negatives = 100 rows total. Balanced.
  - Stages: round_0 and round_4 only (before deliberation kicks in vs after).
  - Layers: all 28.
  - Targets: just y_high_shift (top vs bottom within trial). Continuous shift
    and II are removed for the pilot — y_high_shift is the cleanest signal
    given we're matched-pair within trial.
  - Two probe variants for robustness:
      (a) LogisticRegressionCV  -- L2-regularized logistic regression with
          inner CV to pick C automatically.
      (b) PCA(50) + LogisticRegression -- reduce 3584 -> 50 PCs first,
          then small linear model. Honest at low n.
  - Trial-level GroupKFold so train/val never share a trial.

Outputs:
  - probe_v2_results.parquet
  - probe_v2_summary.parquet
  - probe_v2_curves.png
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================

PARQUET_PATH = Path("/home/haider/Misalignment-Contagion/outputs/activations_pilot/"
                    "pilot_qwen7b_synthetic_fc_20pct_seed42.parquet")
TRIAL_JSONL = Path("/home/haider/Misalignment-Contagion/outputs/primary_em/"
                   "synthetic/qwen-7b-instruct/results.jsonl")

OUT_DIR = Path("/home/haider/Misalignment-Contagion/outputs/probe_results_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAGES = ["round_0", "round_4"]
LAYERS = list(range(28))
N_FOLDS = 5
PCA_DIMS = 50
RANDOM_STATE = 42

# Pilot filter -- only compute outcomes for trials matching the pilot subset.
PILOT_FILTER = dict(
    dataset="synthetic",
    minority_ratio=0.2,
    seed=42,
    model_key="qwen-7b-instruct",
    topology="fc",
)


# =============================================================================
# Outcome computation (filtered to pilot trials)
# =============================================================================

def expected_value(probs: dict[str, float]) -> float:
    return sum(int(k) * v for k, v in probs.items())


def compute_pilot_outcomes(jsonl_path: Path) -> pd.DataFrame:
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if not all(t.get(k) == v for k, v in PILOT_FILTER.items()):
                continue
            for a in t["agents"]:
                if a["role"] != "aligned":
                    continue
                p_base = a["baseline_probs"]
                p_shadow = a.get("shadow_probs")
                if p_shadow is None:
                    continue
                ev_base = expected_value(p_base)
                ev_shadow = expected_value(p_shadow)
                shift = ev_shadow - ev_base
                rows.append({
                    "trial_id": t["trial_id"],
                    "agent_id": a["agent_id"],
                    "ev_baseline": ev_base,
                    "ev_shadow": ev_shadow,
                    "shift": shift,
                    "abs_shift": abs(shift),
                })
    df = pd.DataFrame(rows)
    print(f"Pilot outcomes: {len(df)} aligned agents from {df.trial_id.nunique()} trials")
    print(f"  abs_shift mean/std: {df['abs_shift'].mean():.3f} / {df['abs_shift'].std():.3f}")
    return df


def select_top_bottom_per_trial(df: pd.DataFrame) -> pd.DataFrame:
    """For each trial, pick the agent with max |shift| and the agent with min |shift|.
    Label them y_high_shift = 1 / 0 respectively."""
    rows = []
    for tid, group in df.groupby("trial_id"):
        if len(group) < 2:
            continue
        sorted_g = group.sort_values("abs_shift", ascending=False)
        top = sorted_g.iloc[0].copy()
        bot = sorted_g.iloc[-1].copy()
        if top["abs_shift"] == bot["abs_shift"]:
            # Degenerate trial -- everyone shifted the same amount; skip
            continue
        top["y_high_shift"] = 1
        bot["y_high_shift"] = 0
        rows.append(top)
        rows.append(bot)
    out = pd.DataFrame(rows)
    print(f"After top/bottom selection: {len(out)} rows from {out.trial_id.nunique()} trials")
    print(f"  pos abs_shift mean/std: "
          f"{out[out.y_high_shift==1]['abs_shift'].mean():.3f} / "
          f"{out[out.y_high_shift==1]['abs_shift'].std():.3f}")
    print(f"  neg abs_shift mean/std: "
          f"{out[out.y_high_shift==0]['abs_shift'].mean():.3f} / "
          f"{out[out.y_high_shift==0]['abs_shift'].std():.3f}")
    print(f"  contrast (pos − neg) mean: "
          f"{out[out.y_high_shift==1]['abs_shift'].mean() - out[out.y_high_shift==0]['abs_shift'].mean():.3f}")
    return out


# =============================================================================
# Activation loading
# =============================================================================

def load_activations_for_stage(parquet_path: Path, stage: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, filters=[("stage", "==", stage)])
    df["activation"] = df["activation_bytes"].apply(
        lambda b: np.frombuffer(b, dtype=np.float16).astype(np.float32)
    )
    return df[["trial_id", "agent_id", "layer", "activation"]]


def build_xy(act_df: pd.DataFrame, contrast: pd.DataFrame, layer: int):
    sub = act_df[act_df.layer == layer]
    merged = sub.merge(contrast, on=["trial_id", "agent_id"], how="inner")
    X = np.stack(merged["activation"].values)
    y = merged["y_high_shift"].values.astype(int)
    groups = merged["trial_id"].values
    return X, y, groups


# =============================================================================
# Probe variants
# =============================================================================

def cv_score_l2(X, y, groups, n_folds=N_FOLDS, seed=RANDOM_STATE):
    """L2-regularized LR with inner CV to pick C."""
    gkf = GroupKFold(n_splits=n_folds)
    aucs = []
    for tr, va in gkf.split(X, y, groups):
        if len(np.unique(y[va])) < 2:
            aucs.append(np.nan)
            continue
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
        clf = LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1.0, 10.0],
            cv=3, scoring="roc_auc", max_iter=2000,
            solver="liblinear", random_state=seed,
        )
        clf.fit(Xtr, y[tr])
        scores = clf.decision_function(Xva)
        aucs.append(roc_auc_score(y[va], scores))
    return np.array(aucs)


def cv_score_pca(X, y, groups, n_folds=N_FOLDS, seed=RANDOM_STATE, n_pcs=PCA_DIMS):
    """PCA(50) + small logistic regression."""
    gkf = GroupKFold(n_splits=n_folds)
    aucs = []
    for tr, va in gkf.split(X, y, groups):
        if len(np.unique(y[va])) < 2:
            aucs.append(np.nan)
            continue
        # Cap n_pcs at min(n_train, n_features)
        actual_pcs = min(n_pcs, len(tr) - 1, X.shape[1])
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=actual_pcs, random_state=seed)),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, solver="liblinear",
                                        random_state=seed)),
        ])
        pipe.fit(X[tr], y[tr])
        scores = pipe.decision_function(X[va])
        aucs.append(roc_auc_score(y[va], scores))
    return np.array(aucs)


# =============================================================================
# Main
# =============================================================================

def main():
    raw = compute_pilot_outcomes(TRIAL_JSONL)
    contrast = select_top_bottom_per_trial(raw)
    contrast.to_parquet(OUT_DIR / "contrast_pairs.parquet")

    rows = []
    for stage in STAGES:
        print(f"\n=== Stage: {stage} ===")
        act_df = load_activations_for_stage(PARQUET_PATH, stage)
        for layer in tqdm(LAYERS, desc=stage):
            X, y, groups = build_xy(act_df, contrast, layer)
            # Variant A: full-dim L2 with inner-CV C
            aucs_l2 = cv_score_l2(X, y, groups)
            for fold_idx, s in enumerate(aucs_l2):
                rows.append({"stage": stage, "layer": layer, "variant": "L2_CV",
                             "fold": fold_idx, "auc": s})
            # Variant B: PCA(50) + small LR
            aucs_pca = cv_score_pca(X, y, groups)
            for fold_idx, s in enumerate(aucs_pca):
                rows.append({"stage": stage, "layer": layer, "variant": "PCA50",
                             "fold": fold_idx, "auc": s})

    results = pd.DataFrame(rows)
    results.to_parquet(OUT_DIR / "probe_v2_results.parquet")

    summary = (
        results.groupby(["stage", "layer", "variant"])
        .agg(mean_auc=("auc", "mean"),
             se_auc=("auc", lambda s: s.std(ddof=1) / np.sqrt(s.notna().sum())))
        .reset_index()
    )
    summary.to_parquet(OUT_DIR / "probe_v2_summary.parquet")

    print("\n=== Peak layer per (stage, variant) ===")
    peaks = (summary.sort_values("mean_auc", ascending=False)
                    .groupby(["stage", "variant"]).head(1))
    print(peaks.sort_values(["stage", "variant"])[["stage", "variant", "layer", "mean_auc", "se_auc"]]
          .to_string(index=False))

    plot_curves(summary, OUT_DIR / "probe_v2_curves.png")
    print(f"\nSaved figure -> {OUT_DIR / 'probe_v2_curves.png'}")


def plot_curves(summary: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    cmap = {"L2_CV": "#1f77b4", "PCA50": "#d62728"}
    for ax, stage in zip(axes, STAGES):
        sub = summary[summary.stage == stage]
        for variant in ["L2_CV", "PCA50"]:
            s = sub[sub.variant == variant].sort_values("layer")
            ax.errorbar(
                s["layer"], s["mean_auc"], yerr=s["se_auc"],
                label=variant, color=cmap[variant],
                capsize=2, marker="o", markersize=4, linewidth=1.4,
            )
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="chance")
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUC (top vs bottom shifter within trial)")
        ax.set_title(f"Stage: {stage}")
        ax.set_ylim(0.3, 1.0)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Probe v2: within-trial contrast (Qwen-7B, synthetic, FC, 20%, seed 42)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()