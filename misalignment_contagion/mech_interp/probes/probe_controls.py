# """
# Probe v3 controls.

# Two controls to disentangle scenario-level vs agent-level signal:

#   Control A — Scenario-only baseline for y_shifted.
#     Predict y_shifted from one-hot scenario_id only (no activations).
#     If AUC ≈ 0.70 (matching the activation-based result), the y_shifted
#     signal is entirely scenario-level. If AUC ≈ 0.55, then activations
#     carry real agent-level signal beyond what scenario alone provides.

#   Control B — Within-trial II contrast for internalization.
#     Per trial, take the agent with max II (positive class) and min II
#     (negative class). Run the same layer-wise probe. If round_4 late
#     layers still show high AUC, the internalization signal survives
#     within-scenario matching and is genuinely agent-level.

# Outputs:
#   - control_A_scenario_baseline.txt    (one number per target)
#   - control_B_within_trial_ii.parquet  (full probe sweep)
#   - control_B_curves.png
# """

# from __future__ import annotations

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
# warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
# warnings.filterwarnings("ignore", message="lbfgs failed to converge")

# import json
# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import GroupKFold
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from tqdm import tqdm


# # =============================================================================
# # CONFIG
# # =============================================================================

# OUT_DIR = Path("/home/haider/Misalignment-Contagion/outputs/probe_results_v3")
# PARQUET_PATH = Path(
#     "/home/haider/Misalignment-Contagion/outputs/activations_pilot/"
#     "pilot_qwen7b_moralstories_fc_20pct_400trials.parquet"
# )

# # Reuse the outcomes already computed by probe_sweep_v3.py
# OUTCOMES_PATH = OUT_DIR / "probe_v3_outcomes.parquet"

# STAGES = ["round_0", "round_2", "round_4"]
# LAYERS = list(range(28))
# N_FOLDS = 5
# PCA_DIMS = 50
# RANDOM_STATE = 42


# # =============================================================================
# # Control A: scenario-only baseline
# # =============================================================================

# def control_a_scenario_baseline(outcomes: pd.DataFrame) -> dict:
#     """For each target, train logistic regression on one-hot scenario_id only.
#     Returns the cross-validated AUC for each target.
#     """
#     print("\n=== Control A: scenario-only baseline (no activations) ===")
#     # Extract scenario_id from trial_id (everything before the first '_')
#     outcomes = outcomes.copy()
#     outcomes["scenario_id"] = outcomes["trial_id"].apply(lambda t: t.split("_")[0])

#     n_scenarios = outcomes["scenario_id"].nunique()
#     print(f"  {n_scenarios} unique scenarios across {len(outcomes)} agent rows")

#     # One-hot encode scenarios — this becomes our X
#     X = pd.get_dummies(outcomes["scenario_id"]).values.astype(np.float32)
#     groups = outcomes["trial_id"].values

#     out = {}
#     for target in ["y_shifted", "y_internalized"]:
#         y = outcomes[target].values.astype(int)
#         gkf = GroupKFold(n_splits=N_FOLDS)
#         aucs = []
#         for tr, va in gkf.split(X, y, groups):
#             if len(np.unique(y[va])) < 2 or len(np.unique(y[tr])) < 2:
#                 aucs.append(np.nan)
#                 continue
#             # No scaling needed for one-hot; weak L2 regularization is fine
#             clf = LogisticRegression(
#                 C=1.0, max_iter=2000, solver="liblinear",
#                 random_state=RANDOM_STATE,
#             )
#             clf.fit(X[tr], y[tr])
#             scores = clf.decision_function(X[va])
#             aucs.append(roc_auc_score(y[va], scores))
#         aucs = np.array(aucs)
#         mean = np.nanmean(aucs)
#         se = np.nanstd(aucs, ddof=1) / np.sqrt(np.sum(~np.isnan(aucs)))
#         out[target] = (mean, se)
#         print(f"  {target}: AUC = {mean:.3f} ± {se:.3f}")

#     return out


# # =============================================================================
# # Control B: within-trial II contrast
# # =============================================================================

# def add_within_trial_ii_contrast(outcomes: pd.DataFrame) -> pd.DataFrame:
#     """Add y_high_ii column: 1 if agent has max II in trial, 0 if min, -1 otherwise."""
#     df = outcomes.copy()
#     df["y_high_ii"] = -1
#     n_pairs = 0
#     for tid, group in df.groupby("trial_id"):
#         valid = group.dropna(subset=["ii"])
#         if len(valid) < 2:
#             continue
#         sorted_g = valid.sort_values("ii", ascending=False)
#         top_idx = sorted_g.index[0]
#         bot_idx = sorted_g.index[-1]
#         if df.loc[top_idx, "ii"] == df.loc[bot_idx, "ii"]:
#             continue
#         df.loc[top_idx, "y_high_ii"] = 1
#         df.loc[bot_idx, "y_high_ii"] = 0
#         n_pairs += 1
#     print(f"  within-trial II pairs: {n_pairs} ({n_pairs * 2} rows)")
#     return df


# def load_activations_for_stage(parquet_path: Path, stage: str) -> pd.DataFrame:
#     df = pd.read_parquet(parquet_path, filters=[("stage", "==", stage)])
#     df["activation"] = df["activation_bytes"].apply(
#         lambda b: np.frombuffer(b, dtype=np.float16).astype(np.float32)
#     )
#     return df[["trial_id", "agent_id", "layer", "activation"]]


# def build_xy(act_df, outcomes, layer):
#     sub = act_df[act_df.layer == layer]
#     merged = sub.merge(outcomes, on=["trial_id", "agent_id"], how="inner")
#     merged = merged[merged["y_high_ii"] != -1]
#     X = np.stack(merged["activation"].values)
#     y = merged["y_high_ii"].values.astype(int)
#     groups = merged["trial_id"].values
#     return X, y, groups


# def cv_score_l2(X, y, groups):
#     gkf = GroupKFold(n_splits=N_FOLDS)
#     aucs = []
#     for tr, va in gkf.split(X, y, groups):
#         if len(np.unique(y[va])) < 2 or len(np.unique(y[tr])) < 2:
#             aucs.append(np.nan)
#             continue
#         scaler = StandardScaler().fit(X[tr])
#         Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
#         clf = LogisticRegressionCV(
#             Cs=[0.001, 0.01, 0.1, 1.0, 10.0],
#             cv=3, scoring="roc_auc", max_iter=2000,
#             solver="liblinear", random_state=RANDOM_STATE,
#         )
#         clf.fit(Xtr, y[tr])
#         scores = clf.decision_function(Xva)
#         aucs.append(roc_auc_score(y[va], scores))
#     return np.array(aucs)


# def cv_score_pca(X, y, groups):
#     gkf = GroupKFold(n_splits=N_FOLDS)
#     aucs = []
#     for tr, va in gkf.split(X, y, groups):
#         if len(np.unique(y[va])) < 2 or len(np.unique(y[tr])) < 2:
#             aucs.append(np.nan)
#             continue
#         actual_pcs = min(PCA_DIMS, len(tr) - 1, X.shape[1])
#         pipe = Pipeline([
#             ("scale", StandardScaler()),
#             ("pca", PCA(n_components=actual_pcs, random_state=RANDOM_STATE)),
#             ("clf", LogisticRegression(C=1.0, max_iter=2000,
#                                         solver="liblinear",
#                                         random_state=RANDOM_STATE)),
#         ])
#         pipe.fit(X[tr], y[tr])
#         scores = pipe.decision_function(X[va])
#         aucs.append(roc_auc_score(y[va], scores))
#     return np.array(aucs)


# def control_b_within_trial_ii(outcomes: pd.DataFrame) -> pd.DataFrame:
#     """Within-trial II contrast: same probe sweep as v3, but on y_high_ii only."""
#     print("\n=== Control B: within-trial II contrast ===")
#     contrast = add_within_trial_ii_contrast(outcomes)

#     rows = []
#     for stage in STAGES:
#         print(f"\n  Stage: {stage}")
#         act_df = load_activations_for_stage(PARQUET_PATH, stage)
#         for layer in tqdm(LAYERS, desc=f"  {stage}"):
#             X, y, groups = build_xy(act_df, contrast, layer)
#             if len(y) < 20 or len(np.unique(y)) < 2:
#                 continue
#             for fold_idx, s in enumerate(cv_score_l2(X, y, groups)):
#                 rows.append({"stage": stage, "layer": layer, "variant": "L2_CV",
#                              "fold": fold_idx, "auc": s})
#             for fold_idx, s in enumerate(cv_score_pca(X, y, groups)):
#                 rows.append({"stage": stage, "layer": layer, "variant": "PCA50",
#                              "fold": fold_idx, "auc": s})

#     results = pd.DataFrame(rows)
#     summary = (
#         results.groupby(["stage", "layer", "variant"])
#         .agg(mean_auc=("auc", "mean"),
#              se_auc=("auc", lambda s: s.std(ddof=1) / np.sqrt(s.notna().sum())))
#         .reset_index()
#     )
#     return summary


# def plot_control_b(summary: pd.DataFrame, out_path: Path):
#     fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), sharey=True)
#     cmap = {"L2_CV": "#1f77b4", "PCA50": "#d62728"}
#     for ax, stage in zip(axes, STAGES):
#         sub = summary[summary.stage == stage]
#         for variant in ["L2_CV", "PCA50"]:
#             s = sub[sub.variant == variant].sort_values("layer")
#             if s.empty:
#                 continue
#             ax.errorbar(
#                 s["layer"], s["mean_auc"], yerr=s["se_auc"],
#                 label=variant, color=cmap[variant],
#                 capsize=2, marker="o", markersize=3, linewidth=1.2,
#             )
#         ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
#         ax.set_xlabel("Layer")
#         ax.set_ylabel("AUC (high-II vs low-II within trial)")
#         ax.set_title(f"Stage: {stage}")
#         ax.set_ylim(0.3, 1.0)
#         ax.legend(loc="lower right", fontsize=9)
#         ax.grid(True, alpha=0.3)
#     fig.suptitle(
#         "Control B: within-trial II contrast — does internalization signal "
#         "survive within-scenario matching?",
#         fontsize=11,
#     )
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=150, bbox_inches="tight")
#     plt.close(fig)


# # =============================================================================
# # Main
# # =============================================================================

# def main():
#     if not OUTCOMES_PATH.exists():
#         raise SystemExit(
#             f"Outcomes file not found at {OUTCOMES_PATH}. "
#             "Run probe_sweep_v3.py first."
#         )
#     outcomes = pd.read_parquet(OUTCOMES_PATH)
#     print(f"Loaded {len(outcomes)} outcome rows from {OUTCOMES_PATH.name}")

#     # Control A
#     control_a_results = control_a_scenario_baseline(outcomes)
#     with open(OUT_DIR / "control_A_scenario_baseline.txt", "w") as f:
#         f.write("Control A: scenario-only baseline (one-hot scenario_id, no activations)\n")
#         f.write("=" * 72 + "\n\n")
#         for target, (mean, se) in control_a_results.items():
#             f.write(f"  {target}: AUC = {mean:.3f} ± {se:.3f}\n")
#     print(f"\n  -> saved to {OUT_DIR / 'control_A_scenario_baseline.txt'}")

#     # Control B
#     control_b_summary = control_b_within_trial_ii(outcomes)
#     control_b_summary.to_parquet(OUT_DIR / "control_B_within_trial_ii.parquet")

#     print("\n=== Control B: peak layer per (stage, variant) ===")
#     peaks = (control_b_summary.sort_values("mean_auc", ascending=False)
#                               .groupby(["stage", "variant"]).head(1))
#     print(peaks.sort_values(["stage", "variant"])[
#         ["stage", "variant", "layer", "mean_auc", "se_auc"]
#     ].to_string(index=False))

#     plot_control_b(control_b_summary, OUT_DIR / "control_B_curves.png")
#     print(f"\n  -> saved figure to {OUT_DIR / 'control_B_curves.png'}")

#     # Summary print
#     print("\n" + "=" * 72)
#     print("INTERPRETATION GUIDE")
#     print("=" * 72)
#     a_shifted = control_a_results['y_shifted'][0]
#     a_internalized = control_a_results['y_internalized'][0]
#     print(f"\nControl A (scenario-only baseline):")
#     print(f"  y_shifted:      AUC = {a_shifted:.3f}")
#     print(f"  y_internalized: AUC = {a_internalized:.3f}")
#     print(f"\nv3 main result reported:")
#     print(f"  y_shifted round_0 layer 22:      AUC ≈ 0.70")
#     print(f"  y_internalized round_4 layer 26: AUC ≈ 0.73")
#     print(f"\nIf v3 AUC ≫ Control A AUC → real agent-level signal beyond scenario.")
#     print(f"If v3 AUC ≈ Control A AUC → mostly scenario-level confound.")
#     print(f"\nControl B peak AUC for round_4 (any variant) tells you whether the")
#     print(f"internalization signal survives within-scenario matching.")


# if __name__ == "__main__":
#     main()

"""
Probe v3 controls.

Two controls to disentangle scenario-level vs agent-level signal:

  Control A — Scenario-only baseline for y_shifted.
    Predict y_shifted from one-hot scenario_id only (no activations).
    If AUC ≈ 0.70 (matching the activation-based result), the y_shifted
    signal is entirely scenario-level. If AUC ≈ 0.55, then activations
    carry real agent-level signal beyond what scenario alone provides.

  Control B — Within-trial II contrast for internalization.
    Per trial, take the agent with max II (positive class) and min II
    (negative class). Run the same layer-wise probe. If round_4 late
    layers still show high AUC, the internalization signal survives
    within-scenario matching and is genuinely agent-level.

Outputs:
  - control_A_scenario_baseline.txt    (one number per target)
  - control_B_within_trial_ii.parquet  (full probe sweep)
  - control_B_curves.png
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="lbfgs failed to converge")

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
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

OUT_DIR = Path("/home/haider/Misalignment-Contagion/outputs/probe_results_v3")
PARQUET_PATH = Path(
    "/home/haider/Misalignment-Contagion/outputs/activations_pilot/"
    "pilot_qwen7b_moralstories_fc_20pct_400trials.parquet"
)

# Reuse the outcomes already computed by probe_sweep_v3.py
OUTCOMES_PATH = OUT_DIR / "probe_v3_outcomes.parquet"

STAGES = ["round_0", "round_2", "round_4"]
LAYERS = list(range(28))
N_FOLDS = 5
PCA_DIMS = 50
RANDOM_STATE = 42


# =============================================================================
# Control A: scenario-only baseline
# =============================================================================

def control_a_scenario_baseline(outcomes: pd.DataFrame) -> dict:
    """For each target, train logistic regression on one-hot scenario_id only.
    Returns the cross-validated AUC for each target.
    """
    print("\n=== Control A: scenario-only baseline (no activations) ===")
    # Extract scenario_id from trial_id (everything before the first '_')
    outcomes = outcomes.copy()
    outcomes["scenario_id"] = outcomes["trial_id"].apply(lambda t: t.split("_")[0])

    n_scenarios = outcomes["scenario_id"].nunique()
    print(f"  {n_scenarios} unique scenarios across {len(outcomes)} agent rows")

    # One-hot encode scenarios — this becomes our X
    X = pd.get_dummies(outcomes["scenario_id"]).values.astype(np.float32)
    groups = outcomes["trial_id"].values

    out = {}
    for target in ["y_shifted", "y_internalized"]:
        y = outcomes[target].values.astype(int)
        gkf = GroupKFold(n_splits=N_FOLDS)
        aucs = []
        for tr, va in gkf.split(X, y, groups):
            if len(np.unique(y[va])) < 2 or len(np.unique(y[tr])) < 2:
                aucs.append(np.nan)
                continue
            # No scaling needed for one-hot; weak L2 regularization is fine
            clf = LogisticRegression(
                C=1.0, max_iter=2000, solver="liblinear",
                random_state=RANDOM_STATE,
            )
            clf.fit(X[tr], y[tr])
            scores = clf.decision_function(X[va])
            aucs.append(roc_auc_score(y[va], scores))
        aucs = np.array(aucs)
        mean = np.nanmean(aucs)
        se = np.nanstd(aucs, ddof=1) / np.sqrt(np.sum(~np.isnan(aucs)))
        out[target] = (mean, se)
        print(f"  {target}: AUC = {mean:.3f} ± {se:.3f}")

    return out


# =============================================================================
# Control B: within-trial II contrast
# =============================================================================

def add_within_trial_ii_contrast(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Add y_high_ii column: 1 if agent has max II in trial, 0 if min, -1 otherwise."""
    df = outcomes.copy()
    df["y_high_ii"] = -1
    n_pairs = 0
    for tid, group in df.groupby("trial_id"):
        valid = group.dropna(subset=["ii"])
        if len(valid) < 2:
            continue
        sorted_g = valid.sort_values("ii", ascending=False)
        top_idx = sorted_g.index[0]
        bot_idx = sorted_g.index[-1]
        if df.loc[top_idx, "ii"] == df.loc[bot_idx, "ii"]:
            continue
        df.loc[top_idx, "y_high_ii"] = 1
        df.loc[bot_idx, "y_high_ii"] = 0
        n_pairs += 1
    print(f"  within-trial II pairs: {n_pairs} ({n_pairs * 2} rows)")
    return df


def load_activations_for_stage(parquet_path: Path, stage: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, filters=[("stage", "==", stage)])
    df["activation"] = df["activation_bytes"].apply(
        lambda b: np.frombuffer(b, dtype=np.float16).astype(np.float32)
    )
    return df[["trial_id", "agent_id", "layer", "activation"]]


def build_xy(act_df, outcomes, layer):
    sub = act_df[act_df.layer == layer]
    merged = sub.merge(outcomes, on=["trial_id", "agent_id"], how="inner")
    merged = merged[merged["y_high_ii"] != -1]
    X = np.stack(merged["activation"].values)
    y = merged["y_high_ii"].values.astype(int)
    groups = merged["trial_id"].values
    return X, y, groups


def _fit_one_l2_fold(X_tr, y_tr, X_va, y_va, seed):
    """Fit one fold of L2 logistic regression. Returns AUC or nan."""
    if len(np.unique(y_va)) < 2 or len(np.unique(y_tr)) < 2:
        return np.nan
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)
    # lbfgs is multi-threaded internally and faster than liblinear at this scale.
    clf = LogisticRegression(
        C=0.1, max_iter=5000, solver="lbfgs",
        random_state=seed,
    )
    clf.fit(X_tr_s, y_tr)
    return roc_auc_score(y_va, clf.decision_function(X_va_s))


def cv_score_l2(X, y, groups):
    """Plain L2 logistic regression at fixed C=0.1, folds run in parallel."""
    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = list(gkf.split(X, y, groups))
    aucs = Parallel(n_jobs=N_FOLDS, backend="loky")(
        delayed(_fit_one_l2_fold)(
            X[tr], y[tr], X[va], y[va], RANDOM_STATE
        )
        for tr, va in folds
    )
    return np.array(aucs)


def _fit_one_pca_fold(X_tr, y_tr, X_va, y_va, seed, n_pcs):
    """Fit one fold of PCA + small logistic regression. Returns AUC or nan."""
    if len(np.unique(y_va)) < 2 or len(np.unique(y_tr)) < 2:
        return np.nan
    actual_pcs = min(n_pcs, len(X_tr) - 1, X_tr.shape[1])
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=actual_pcs, random_state=seed)),
        ("clf", LogisticRegression(
            C=1.0, max_iter=5000, solver="lbfgs", random_state=seed,
        )),
    ])
    pipe.fit(X_tr, y_tr)
    return roc_auc_score(y_va, pipe.decision_function(X_va))


def cv_score_pca(X, y, groups):
    """PCA(50) + small LR, folds run in parallel."""
    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = list(gkf.split(X, y, groups))
    aucs = Parallel(n_jobs=N_FOLDS, backend="loky")(
        delayed(_fit_one_pca_fold)(
            X[tr], y[tr], X[va], y[va], RANDOM_STATE, PCA_DIMS
        )
        for tr, va in folds
    )
    return np.array(aucs)


def control_b_within_trial_ii(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Within-trial II contrast: same probe sweep as v3, but on y_high_ii only."""
    print("\n=== Control B: within-trial II contrast ===")
    contrast = add_within_trial_ii_contrast(outcomes)

    rows = []
    for stage in STAGES:
        print(f"\n  Stage: {stage}")
        act_df = load_activations_for_stage(PARQUET_PATH, stage)
        for layer in tqdm(LAYERS, desc=f"  {stage}"):
            X, y, groups = build_xy(act_df, contrast, layer)
            if len(y) < 20 or len(np.unique(y)) < 2:
                continue
            for fold_idx, s in enumerate(cv_score_l2(X, y, groups)):
                rows.append({"stage": stage, "layer": layer, "variant": "L2_CV",
                             "fold": fold_idx, "auc": s})
            for fold_idx, s in enumerate(cv_score_pca(X, y, groups)):
                rows.append({"stage": stage, "layer": layer, "variant": "PCA50",
                             "fold": fold_idx, "auc": s})

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["stage", "layer", "variant"])
        .agg(mean_auc=("auc", "mean"),
             se_auc=("auc", lambda s: s.std(ddof=1) / np.sqrt(s.notna().sum())))
        .reset_index()
    )
    return summary


def plot_control_b(summary: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), sharey=True)
    cmap = {"L2_CV": "#1f77b4", "PCA50": "#d62728"}
    for ax, stage in zip(axes, STAGES):
        sub = summary[summary.stage == stage]
        for variant in ["L2_CV", "PCA50"]:
            s = sub[sub.variant == variant].sort_values("layer")
            if s.empty:
                continue
            ax.errorbar(
                s["layer"], s["mean_auc"], yerr=s["se_auc"],
                label=variant, color=cmap[variant],
                capsize=2, marker="o", markersize=3, linewidth=1.2,
            )
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUC (high-II vs low-II within trial)")
        ax.set_title(f"Stage: {stage}")
        ax.set_ylim(0.3, 1.0)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Control B: within-trial II contrast — does internalization signal "
        "survive within-scenario matching?",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main(skip_a: bool = False):
    if not OUTCOMES_PATH.exists():
        raise SystemExit(
            f"Outcomes file not found at {OUTCOMES_PATH}. "
            "Run probe_sweep_v3.py first."
        )
    outcomes = pd.read_parquet(OUTCOMES_PATH)
    print(f"Loaded {len(outcomes)} outcome rows from {OUTCOMES_PATH.name}")

    # Control A
    if skip_a:
        print("\n=== Control A: SKIPPED (--skip-a) ===")
        control_a_results = None
    else:
        control_a_results = control_a_scenario_baseline(outcomes)
        with open(OUT_DIR / "control_A_scenario_baseline.txt", "w") as f:
            f.write("Control A: scenario-only baseline (one-hot scenario_id, no activations)\n")
            f.write("=" * 72 + "\n\n")
            for target, (mean, se) in control_a_results.items():
                f.write(f"  {target}: AUC = {mean:.3f} ± {se:.3f}\n")
        print(f"\n  -> saved to {OUT_DIR / 'control_A_scenario_baseline.txt'}")

    # Control B
    control_b_summary = control_b_within_trial_ii(outcomes)
    control_b_summary.to_parquet(OUT_DIR / "control_B_within_trial_ii.parquet")

    print("\n=== Control B: peak layer per (stage, variant) ===")
    peaks = (control_b_summary.sort_values("mean_auc", ascending=False)
                              .groupby(["stage", "variant"]).head(1))
    print(peaks.sort_values(["stage", "variant"])[
        ["stage", "variant", "layer", "mean_auc", "se_auc"]
    ].to_string(index=False))

    plot_control_b(control_b_summary, OUT_DIR / "control_B_curves.png")
    print(f"\n  -> saved figure to {OUT_DIR / 'control_B_curves.png'}")

    # Summary print
    print("\n" + "=" * 72)
    print("INTERPRETATION GUIDE")
    print("=" * 72)
    if control_a_results is not None:
        a_shifted = control_a_results['y_shifted'][0]
        a_internalized = control_a_results['y_internalized'][0]
        print(f"\nControl A (scenario-only baseline):")
        print(f"  y_shifted:      AUC = {a_shifted:.3f}")
        print(f"  y_internalized: AUC = {a_internalized:.3f}")
    print(f"\nv3 main result reported:")
    print(f"  y_shifted round_0 layer 22:      AUC ≈ 0.70")
    print(f"  y_internalized round_4 layer 26: AUC ≈ 0.73")
    print(f"\nControl B peak AUC for round_4 (any variant) tells you whether the")
    print(f"internalization signal survives within-scenario matching.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-a", action="store_true",
                        help="Skip Control A (already saved or known uninformative).")
    args = parser.parse_args()
    main(skip_a=args.skip_a)