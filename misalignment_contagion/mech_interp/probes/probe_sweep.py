"""
Probe sweep v3 — moral_stories pilot.

Design:
  - Data: 400-trial moral_stories pilot (Qwen-7B-Instruct, FC, 20%, seed 42,
    model_induced, rigid:rigid).
  - Probe target: y_shifted (binary, |shadow shift| >= 0.5). With moral_stories
    having ~65% conversion, this gives a real ~65/35 class split (vs synthetic's
    pathological 94/6).
  - Robustness target: y_internalized (binary, II >= 0.7). For comparison.
  - Within-trial top/bottom contrast as a third target for matched-pair
    sanity check.
  - Stages: round_0, round_2, round_4 (early / mid / late deliberation).
  - Layers: all 28.
  - Two probe variants:
      (a) LogisticRegressionCV  -- L2 with inner-CV C selection
      (b) PCA(50) + LogisticRegression  -- low-dim sanity check
  - Trial-level GroupKFold (5 folds) so train/val never share a trial.

Outputs:
  - probe_v3_outcomes.parquet
  - probe_v3_results.parquet           (per-fold)
  - probe_v3_summary.parquet           (mean +/- SE)
  - probe_v3_curves.png                (the headline figure)
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
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

PARQUET_PATH = Path(
    "/home/haider/Misalignment-Contagion/outputs/activations_pilot/"
    "pilot_qwen7b_moralstories_fc_20pct_400trials.parquet"
)
TRIAL_JSONL = Path(
    "/home/haider/Misalignment-Contagion/outputs/primary_em/"
    "moral_stories/qwen-7b-instruct/results.jsonl"
)
OUT_DIR = Path("/home/haider/Misalignment-Contagion/outputs/probe_results_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Match the pilot extraction filter exactly
PILOT_FILTER = dict(
    dataset="moral_stories",
    minority_ratio=0.2,
    model_key="qwen-7b-instruct",
    model_condition="model_induced",
    prompt_strategy="rigid:rigid",
    seed=42,
)
PILOT_TOPOLOGY = "fc"

STAGES = ["round_0", "round_2", "round_4"]
LAYERS = list(range(28))
N_FOLDS = 5
PCA_DIMS = 50
SHIFT_THRESHOLD = 0.5
II_THRESHOLD = 0.7
RANDOM_STATE = 42


# =============================================================================
# Outcome computation
# =============================================================================

def expected_value(probs: dict[str, float]) -> float:
    return sum(int(k) * v for k, v in probs.items())


def jsd(p: dict[str, float], q: dict[str, float]) -> float:
    keys = sorted(set(p) | set(q), key=int)
    pv = np.array([p.get(k, 0.0) for k in keys])
    qv = np.array([q.get(k, 0.0) for k in keys])
    pv = pv / pv.sum()
    qv = qv / qv.sum()
    d = jensenshannon(pv, qv, base=2)
    return float(d ** 2)


def compute_pilot_outcomes(jsonl_path: Path) -> pd.DataFrame:
    """Aligned-agent outcomes for the pilot subset only."""
    rows = []
    n_lines = 0
    n_skipped = 0
    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            n_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError as e:
                n_skipped += 1
                print(f"  WARN: skipping malformed line {line_num} ({e})")
                continue
            if not all(t.get(k) == v for k, v in PILOT_FILTER.items()):
                continue
            if t.get("topology", "").lower() != PILOT_TOPOLOGY:
                continue
            for a in t["agents"]:
                if a["role"] != "aligned":
                    continue
                p_base = a["baseline_probs"]
                p_final = a["round_probs"][-1]
                p_shadow = a.get("shadow_probs")
                if p_shadow is None:
                    continue
                ev_base = expected_value(p_base)
                ev_shadow = expected_value(p_shadow)
                jsd_final = jsd(p_final, p_base)
                jsd_shadow = jsd(p_shadow, p_base)
                ii = (jsd_shadow / jsd_final) if jsd_final > 1e-9 else np.nan
                shift = ev_shadow - ev_base
                rows.append({
                    "trial_id": t["trial_id"],
                    "agent_id": a["agent_id"],
                    "ev_baseline": ev_base,
                    "ev_shadow": ev_shadow,
                    "shift": shift,
                    "abs_shift": abs(shift),
                    "ii": ii,
                    "y_shifted": int(abs(shift) >= SHIFT_THRESHOLD),
                    "y_internalized": int(
                        (not np.isnan(ii)) and ii >= II_THRESHOLD
                    ),
                })
    df = pd.DataFrame(rows)
    print(f"Read {n_lines} lines, skipped {n_skipped} malformed.")
    print(f"Pilot outcomes: {len(df)} aligned-agent rows from {df.trial_id.nunique()} trials")
    print(f"  abs_shift mean/std:    {df['abs_shift'].mean():.3f} / {df['abs_shift'].std():.3f}")
    print(f"  shift mean/std:        {df['shift'].mean():.3f} / {df['shift'].std():.3f}")
    print(f"  II mean/std (non-nan): {df['ii'].dropna().mean():.3f} / {df['ii'].dropna().std():.3f}")
    print(f"  y_shifted=1:           {df['y_shifted'].mean():.3f}  (target: ~0.65 per paper)")
    print(f"  y_internalized=1:      {df['y_internalized'].mean():.3f}")
    return df


def add_within_trial_contrast(df: pd.DataFrame) -> pd.DataFrame:
    """Add y_top_shifter column: 1 if this agent has max |shift| in its trial,
    0 if it has min |shift|, -1 (excluded) otherwise.
    """
    df = df.copy()
    df["y_top_shifter"] = -1
    for tid, group in df.groupby("trial_id"):
        if len(group) < 2:
            continue
        sorted_g = group.sort_values("abs_shift", ascending=False)
        top_idx = sorted_g.index[0]
        bot_idx = sorted_g.index[-1]
        if df.loc[top_idx, "abs_shift"] == df.loc[bot_idx, "abs_shift"]:
            continue
        df.loc[top_idx, "y_top_shifter"] = 1
        df.loc[bot_idx, "y_top_shifter"] = 0
    n_pairs = (df["y_top_shifter"] != -1).sum() // 2
    print(f"  within-trial pairs:    {n_pairs} pairs ({n_pairs * 2} rows)")
    return df


# =============================================================================
# Activation loading
# =============================================================================

def load_activations_for_stage(parquet_path: Path, stage: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, filters=[("stage", "==", stage)])
    df["activation"] = df["activation_bytes"].apply(
        lambda b: np.frombuffer(b, dtype=np.float16).astype(np.float32)
    )
    return df[["trial_id", "agent_id", "layer", "activation"]]


def build_xy(act_df, outcomes, layer, target_col, restrict_to=None):
    """Inner-join activations at one layer with outcomes. Optional restriction
    (for the within-trial contrast: only keep rows with y_top_shifter != -1).
    Returns (X, y, groups)."""
    sub = act_df[act_df.layer == layer]
    merged = sub.merge(outcomes, on=["trial_id", "agent_id"], how="inner")
    if restrict_to is not None:
        merged = merged[merged[restrict_to] != -1]
    X = np.stack(merged["activation"].values)
    y = merged[target_col].values.astype(int)
    groups = merged["trial_id"].values
    return X, y, groups


# =============================================================================
# Probes
# =============================================================================

def cv_score_l2(X, y, groups, n_folds=N_FOLDS, seed=RANDOM_STATE):
    """L2-regularized LR with inner CV to pick C.
    Uses lbfgs solver — much faster than liblinear at n>1000 with thousands of features.
    Inner CV reduced to 2 folds for speed; outer CV still 5 folds.
    """
    gkf = GroupKFold(n_splits=n_folds)
    aucs = []
    for tr, va in gkf.split(X, y, groups):
        if len(np.unique(y[va])) < 2 or len(np.unique(y[tr])) < 2:
            aucs.append(np.nan)
            continue
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
        clf = LogisticRegressionCV(
            Cs=[0.01, 0.1, 1.0, 10.0],
            cv=2, scoring="roc_auc", max_iter=1000,
            solver="lbfgs", random_state=seed, n_jobs=1,
        )
        clf.fit(Xtr, y[tr])
        scores = clf.decision_function(Xva)
        aucs.append(roc_auc_score(y[va], scores))
    return np.array(aucs)


def cv_score_pca(X, y, groups, n_folds=N_FOLDS, seed=RANDOM_STATE, n_pcs=PCA_DIMS):
    """PCA(n_pcs) + small logistic regression."""
    gkf = GroupKFold(n_splits=n_folds)
    aucs = []
    for tr, va in gkf.split(X, y, groups):
        if len(np.unique(y[va])) < 2 or len(np.unique(y[tr])) < 2:
            aucs.append(np.nan)
            continue
        actual_pcs = min(n_pcs, len(tr) - 1, X.shape[1])
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=actual_pcs, random_state=seed)),
            ("clf", LogisticRegression(C=1.0, max_iter=2000,
                                        solver="liblinear", random_state=seed)),
        ])
        pipe.fit(X[tr], y[tr])
        scores = pipe.decision_function(X[va])
        aucs.append(roc_auc_score(y[va], scores))
    return np.array(aucs)


# =============================================================================
# Main
# =============================================================================

TARGETS = [
    # (column_name, restrict_to_column_or_None, display_name)
    ("y_shifted",       None,             "shifted (|Δ|≥0.5)"),
    ("y_internalized",  None,             "internalized (II≥0.7)"),
    ("y_top_shifter",   "y_top_shifter",  "top vs bottom shifter (within trial)"),
]


def main():
    outcomes = compute_pilot_outcomes(TRIAL_JSONL)
    outcomes = add_within_trial_contrast(outcomes)
    outcomes.to_parquet(OUT_DIR / "probe_v3_outcomes.parquet")

    rows = []
    for stage in STAGES:
        print(f"\n=== Stage: {stage} ===")
        act_df = load_activations_for_stage(PARQUET_PATH, stage)
        for layer in tqdm(LAYERS, desc=stage):
            for target_col, restrict_to, _ in TARGETS:
                X, y, groups = build_xy(
                    act_df, outcomes, layer, target_col, restrict_to=restrict_to
                )
                if len(y) < 20 or len(np.unique(y)) < 2:
                    # Skip degenerate cases
                    continue
                # L2-CV
                for fold_idx, s in enumerate(cv_score_l2(X, y, groups)):
                    rows.append({
                        "stage": stage, "layer": layer, "target": target_col,
                        "variant": "L2_CV", "fold": fold_idx, "auc": s,
                    })
                # PCA
                for fold_idx, s in enumerate(cv_score_pca(X, y, groups)):
                    rows.append({
                        "stage": stage, "layer": layer, "target": target_col,
                        "variant": "PCA50", "fold": fold_idx, "auc": s,
                    })

    results = pd.DataFrame(rows)
    results.to_parquet(OUT_DIR / "probe_v3_results.parquet")

    summary = (
        results.groupby(["stage", "layer", "target", "variant"])
        .agg(
            mean_auc=("auc", "mean"),
            se_auc=("auc", lambda s: s.std(ddof=1) / np.sqrt(s.notna().sum())),
        )
        .reset_index()
    )
    summary.to_parquet(OUT_DIR / "probe_v3_summary.parquet")

    print("\n=== Peak layer per (stage, target, variant) ===")
    peaks = (summary.sort_values("mean_auc", ascending=False)
                    .groupby(["stage", "target", "variant"]).head(1))
    print(peaks.sort_values(["target", "stage", "variant"])[
        ["stage", "target", "variant", "layer", "mean_auc", "se_auc"]
    ].to_string(index=False))

    plot_curves(summary, OUT_DIR / "probe_v3_curves.png")
    print(f"\nSaved figure -> {OUT_DIR / 'probe_v3_curves.png'}")


def plot_curves(summary: pd.DataFrame, out_path: Path):
    target_titles = {
        "y_shifted":      "Shifted (|Δ|≥0.5)",
        "y_internalized": "Internalized (II≥0.7)",
        "y_top_shifter":  "Top vs bottom shifter",
    }
    targets = list(target_titles.keys())
    fig, axes = plt.subplots(
        len(targets), len(STAGES),
        figsize=(4.5 * len(STAGES), 3.5 * len(targets)),
        sharey=True, sharex=True,
    )
    cmap = {"L2_CV": "#1f77b4", "PCA50": "#d62728"}
    for i, target in enumerate(targets):
        for j, stage in enumerate(STAGES):
            ax = axes[i, j] if len(targets) > 1 else axes[j]
            sub = summary[(summary.target == target) & (summary.stage == stage)]
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
            ax.set_ylim(0.3, 1.0)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(stage)
            if j == 0:
                ax.set_ylabel(f"{target_titles[target]}\nAUC")
            if i == len(targets) - 1:
                ax.set_xlabel("Layer")
            if i == 0 and j == len(STAGES) - 1:
                ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Probe v3: layer-wise AUC across deliberation stages\n"
        "(Qwen-7B, moral_stories, FC, 20%, seed 42, 400 trials)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()