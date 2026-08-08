"""
Regression: predict aligned-agent shadow shift from graph features.

Loads:
  - per_graph_per_dataset.csv  (105 rows: 21 unique graphs × 5 datasets, outcomes only)
  - unique_graphs_features.csv (21 rows:   graph features per unique graph)

Joins per-dataset outcomes with graph features on (topology, minority_ratio,
position_config). Fits OLS, Ridge, Lasso under two strategies:

  Strategy A — Dataset dummies as features (in-sample interpretation):
    Adds 4 one-hot dataset indicators to the topology features.
    Reports R², coefficients. Dataset dummies absorb dataset main effects;
    topology coefficients reflect within-dataset effects.

  Strategy B — Within-dataset z-scored outcome (out-of-sample LODO CV):
    Z-scores y_mean_shadow_shift within each dataset, then runs leave-one-
    dataset-out CV using topology features only. This is the honest test of
    whether graph structure generalizes across content domains.

Outputs:
  regression_outputs/
    regression_results.csv       — model performance summary (R², MAE, MSE)
    coefficients_strategy_A.csv  — coefficients from in-sample fits
    coefficients_strategy_B.csv  — LODO coefficient stability (selected per fold)
    lodo_predictions.csv         — per-row predicted vs actual under LODO CV
    plots/
      predicted_vs_actual.png    — model predictions vs truth
      coefficient_stability.png  — which features survive LODO folds
      residuals_by_dataset.png   — dataset-level error pattern
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ID_COLS = ["graph_id", "family", "minority_ratio", "dataset"]
PROVENANCE_COLS = ["n_trials", "n_aligned_obs", "placement_summary"]
TARGET_COLS = ["y_mean_shadow_shift", "y_var_shadow_shift", "y_sem_shadow_shift"]
NON_FEATURE_COLS = {"graph_id", "family", "dataset", "placement_summary"}


# ── Data ─────────────────────────────────────────────────────────────

def load_data(per_dataset_path: Path, features_path: Path | None = None) -> pd.DataFrame:
    """Load per-dataset outcomes already joined with graph features.

    The new join_outcomes pipeline produces a single CSV with outcomes + features
    pre-merged on graph_id. features_path is accepted but ignored when None
    (kept for backwards compatibility).
    """
    df = pd.read_csv(per_dataset_path)
    if features_path is not None and Path(features_path).exists():
        df_feat = pd.read_csv(features_path)
        pooled_y = [c for c in TARGET_COLS if c in df_feat.columns]
        pooled_prov = [c for c in PROVENANCE_COLS if c in df_feat.columns]
        df_feat = df_feat.drop(columns=pooled_y + pooled_prov, errors="ignore")
        df = df.merge(df_feat, on="graph_id", how="left", validate="many_to_one")
    assert df["graph_id"].notna().all(), "graph_id missing in some rows"
    return df


def prep_features(df: pd.DataFrame, target: str = "y_mean_shadow_shift",
                  drop_constants: bool = True) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return feature matrix X, target y, and feature names. Drops constants."""
    y = df[target].copy()
    drop = set(ID_COLS) | set(PROVENANCE_COLS) | set(TARGET_COLS) | NON_FEATURE_COLS
    feat_cols = [c for c in df.columns if c not in drop]
    X = df[feat_cols].copy()
    # Drop any remaining non-numeric columns defensively
    non_numeric = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
    if non_numeric:
        X = X.drop(columns=non_numeric)
        feat_cols = [c for c in feat_cols if c not in non_numeric]

    if drop_constants:
        const = [c for c in feat_cols if X[c].nunique() <= 1]
        X = X.drop(columns=const)
        feat_cols = [c for c in feat_cols if c not in const]

    return X, y, feat_cols


# ── Strategy A: in-sample with dataset dummies ────────────────────────

def add_dataset_dummies(X: pd.DataFrame, datasets: pd.Series,
                        reference: str | None = None) -> pd.DataFrame:
    """One-hot encode dataset, drop reference category."""
    dummies = pd.get_dummies(datasets, prefix="dataset", drop_first=False).astype(float)
    if reference is None:
        reference = sorted(datasets.unique())[0]
    ref_col = f"dataset_{reference}"
    if ref_col in dummies.columns:
        dummies = dummies.drop(columns=[ref_col])
    return pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)


def fit_one(X: np.ndarray, y: np.ndarray, model_name: str):
    """Fit a single model on standardized X."""
    if model_name == "linear":
        m = LinearRegression()
    elif model_name == "ridge":
        m = RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5)
    elif model_name == "lasso":
        m = LassoCV(alphas=np.logspace(-3, 1, 50), cv=5, max_iter=20000,
                    n_alphas=50, random_state=42)
    else:
        raise ValueError(model_name)
    m.fit(X, y)
    return m


def strategy_a(df: pd.DataFrame, target: str, out_dir: Path) -> dict:
    """In-sample fit with dataset dummies. Returns metrics + coefficients."""
    X_topo, y, topo_features = prep_features(df, target=target)
    X = add_dataset_dummies(X_topo, df["dataset"])
    all_features = list(X.columns)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X.values)
    y_arr = y.values

    results = []
    coef_rows = []
    preds = {}
    for name in ["linear", "ridge", "lasso"]:
        m = fit_one(X_s, y_arr, name)
        y_hat = m.predict(X_s)
        preds[name] = y_hat
        results.append({
            "strategy": "A_dummies",
            "target": target,
            "model": name,
            "n_features": len(all_features),
            "n_nonzero": int(np.sum(np.abs(m.coef_) > 1e-8)),
            "in_sample_R2": r2_score(y_arr, y_hat),
            "in_sample_MAE": mean_absolute_error(y_arr, y_hat),
            "in_sample_MSE": mean_squared_error(y_arr, y_hat),
            "alpha": getattr(m, "alpha_", None),
        })
        for feat, c in zip(all_features, m.coef_):
            coef_rows.append({
                "strategy": "A_dummies", "target": target, "model": name,
                "feature": feat, "coefficient": float(c),
                "is_dataset_dummy": feat.startswith("dataset_"),
            })

    return {
        "results": pd.DataFrame(results),
        "coefficients": pd.DataFrame(coef_rows),
        "predictions": preds,
        "y_true": y_arr,
        "feature_names": all_features,
        "df_with_meta": df.reset_index(drop=True),
    }


# ── Strategy B: LODO CV with within-dataset z-scoring ─────────────────

def zscore_within(y: pd.Series, group: pd.Series) -> pd.Series:
    """Z-score y within each group (dataset)."""
    out = y.copy().astype(float)
    for g in group.unique():
        mask = group == g
        sub = y[mask]
        out.loc[mask] = (sub - sub.mean()) / sub.std(ddof=0) if sub.std(ddof=0) > 0 else 0.0
    return out


def strategy_b(df: pd.DataFrame, target: str, out_dir: Path) -> dict:
    """LODO CV with within-dataset z-scored outcome and topology features only."""
    X, y_raw, topo_features = prep_features(df, target=target)
    datasets = df["dataset"].values
    y = zscore_within(df[target], df["dataset"]).values

    unique_datasets = sorted(np.unique(datasets))
    results = []
    coef_rows = []
    fold_preds = []

    for name in ["linear", "ridge", "lasso"]:
        all_y_true, all_y_pred, all_meta = [], [], []
        for held_out in unique_datasets:
            train_mask = datasets != held_out
            test_mask = ~train_mask

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_mask].values)
            X_test = scaler.transform(X[test_mask].values)

            m = fit_one(X_train, y[train_mask], name)
            y_pred = m.predict(X_test)

            all_y_true.append(y[test_mask])
            all_y_pred.append(y_pred)
            sub_df = df[test_mask][ID_COLS].copy().reset_index(drop=True)
            sub_df["model"] = name
            sub_df["target"] = target
            sub_df["held_out_dataset"] = held_out
            sub_df["y_true_z"] = y[test_mask]
            sub_df["y_pred_z"] = y_pred
            all_meta.append(sub_df)

            # Per-fold coefficients
            for feat, c in zip(topo_features, m.coef_):
                coef_rows.append({
                    "strategy": "B_lodo", "target": target, "model": name,
                    "held_out_dataset": held_out,
                    "feature": feat, "coefficient": float(c),
                    "selected": abs(c) > 1e-8,
                })

        yt = np.concatenate(all_y_true)
        yp = np.concatenate(all_y_pred)
        results.append({
            "strategy": "B_lodo",
            "target": target,
            "model": name,
            "n_features": len(topo_features),
            "lodo_R2": r2_score(yt, yp),
            "lodo_MAE": mean_absolute_error(yt, yp),
            "lodo_MSE": mean_squared_error(yt, yp),
        })
        fold_preds.append(pd.concat(all_meta, ignore_index=True))

    return {
        "results": pd.DataFrame(results),
        "coefficients": pd.DataFrame(coef_rows),
        "predictions": pd.concat(fold_preds, ignore_index=True),
        "feature_names": topo_features,
    }


# ── Strategy C: random k-fold CV on mixed rows ───────────────────────

def strategy_c(df: pd.DataFrame, target: str, out_dir: Path,
               n_splits: int = 5, seed: int = 42) -> dict:
    """K-fold random CV on (graph, dataset) rows.

    Features = topology + dataset dummies. Target = raw y. Tests in-distribution
    predictive accuracy: can topology + dataset identity predict held-out cells
    when the model sees both datasets in training?
    """
    from sklearn.model_selection import KFold

    X_topo, y, topo_features = prep_features(df, target=target)
    X = add_dataset_dummies(X_topo, df["dataset"])
    all_features = list(X.columns)
    X_arr = X.values
    y_arr = y.values

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = []
    coef_rows = []
    fold_preds = []

    for name in ["linear", "ridge", "lasso"]:
        all_y_true, all_y_pred, all_meta = [], [], []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_arr)):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_arr[train_idx])
            X_test = scaler.transform(X_arr[test_idx])

            m = fit_one(X_train, y_arr[train_idx], name)
            y_pred = m.predict(X_test)

            all_y_true.append(y_arr[test_idx])
            all_y_pred.append(y_pred)

            sub_df = df.iloc[test_idx][ID_COLS].copy().reset_index(drop=True)
            sub_df["model"] = name
            sub_df["target"] = target
            sub_df["fold"] = fold_idx
            sub_df["y_true"] = y_arr[test_idx]
            sub_df["y_pred"] = y_pred
            all_meta.append(sub_df)

            for feat, c in zip(all_features, m.coef_):
                coef_rows.append({
                    "strategy": "C_kfold", "target": target, "model": name,
                    "fold": fold_idx,
                    "feature": feat, "coefficient": float(c),
                    "is_dataset_dummy": feat.startswith("dataset_"),
                })

        yt = np.concatenate(all_y_true)
        yp = np.concatenate(all_y_pred)
        results.append({
            "strategy": "C_kfold",
            "target": target,
            "model": name,
            "n_features": len(all_features),
            "n_splits": n_splits,
            "cv_R2": r2_score(yt, yp),
            "cv_MAE": mean_absolute_error(yt, yp),
            "cv_MSE": mean_squared_error(yt, yp),
        })
        fold_preds.append(pd.concat(all_meta, ignore_index=True))

    return {
        "results": pd.DataFrame(results),
        "coefficients": pd.DataFrame(coef_rows),
        "predictions": pd.concat(fold_preds, ignore_index=True),
        "feature_names": all_features,
    }


# ── Strategy D: baseline (no topology, only dataset + minority_ratio) ─

def strategy_d(df: pd.DataFrame, target: str, out_dir: Path,
               n_splits: int = 5, seed: int = 42) -> dict:
    """5-fold CV baseline: only dataset dummies + minority_ratio as features.

    Headline comparison: does adding topology features beat this trivial
    baseline? If Strategy C R^2 ~= Strategy D R^2, topology is doing nothing.
    """
    from sklearn.model_selection import KFold

    y = df[target].values
    dummies = pd.get_dummies(df["dataset"], prefix="dataset",
                              drop_first=True).astype(float)
    X = pd.concat([df[["minority_ratio"]].reset_index(drop=True),
                   dummies.reset_index(drop=True)], axis=1)
    all_features = list(X.columns)
    X_arr = X.values

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = []
    fold_preds = []

    for name in ["linear", "ridge", "lasso"]:
        all_y_true, all_y_pred, all_meta = [], [], []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_arr)):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_arr[train_idx])
            X_test = scaler.transform(X_arr[test_idx])
            m = fit_one(X_train, y[train_idx], name)
            y_pred = m.predict(X_test)
            all_y_true.append(y[test_idx])
            all_y_pred.append(y_pred)
            sub = df.iloc[test_idx][ID_COLS].copy().reset_index(drop=True)
            sub["model"] = name
            sub["target"] = target
            sub["fold"] = fold_idx
            sub["y_true"] = y[test_idx]
            sub["y_pred"] = y_pred
            all_meta.append(sub)

        yt = np.concatenate(all_y_true)
        yp = np.concatenate(all_y_pred)
        results.append({
            "strategy": "D_baseline",
            "target": target,
            "model": name,
            "n_features": len(all_features),
            "n_splits": n_splits,
            "cv_R2": r2_score(yt, yp),
            "cv_MAE": mean_absolute_error(yt, yp),
            "cv_MSE": mean_squared_error(yt, yp),
        })
        fold_preds.append(pd.concat(all_meta, ignore_index=True))

    return {
        "results": pd.DataFrame(results),
        "predictions": pd.concat(fold_preds, ignore_index=True),
        "feature_names": all_features,
    }


# ── Plots ────────────────────────────────────────────────────────────

def plot_predicted_vs_actual(a_out: dict, b_out: dict, target: str, path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for j, name in enumerate(["linear", "ridge", "lasso"]):
        # Strategy A
        ax = axes[0, j]
        yt, yp = a_out["y_true"], a_out["predictions"][name]
        ax.scatter(yt, yp, alpha=0.6, s=30)
        lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lim, lim, "k--", alpha=0.4, linewidth=1)
        ax.set_xlabel(f"actual {target}")
        ax.set_ylabel(f"predicted {target}")
        r2 = r2_score(yt, yp)
        ax.set_title(f"A (dummies) — {name}\nR² = {r2:.3f}")
        ax.grid(alpha=0.3)

        # Strategy B
        ax = axes[1, j]
        sub = b_out["predictions"][b_out["predictions"]["model"] == name]
        ax.scatter(sub["y_true_z"], sub["y_pred_z"], alpha=0.6, s=30,
                   c=pd.Categorical(sub["held_out_dataset"]).codes, cmap="tab10")
        lim = [min(sub["y_true_z"].min(), sub["y_pred_z"].min()),
               max(sub["y_true_z"].max(), sub["y_pred_z"].max())]
        ax.plot(lim, lim, "k--", alpha=0.4, linewidth=1)
        ax.set_xlabel("actual z-scored y")
        ax.set_ylabel("predicted z-scored y")
        r2 = r2_score(sub["y_true_z"], sub["y_pred_z"])
        ax.set_title(f"B (LODO) — {name}\nR² = {r2:.3f}")
        ax.grid(alpha=0.3)

    fig.suptitle(f"Predicted vs Actual — {target}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_coef_stability(b_out: dict, target: str, path: Path, top_k: int = 20):
    """Show which features get nonzero coefficients across the 5 LODO folds (Lasso)."""
    coefs = b_out["coefficients"]
    coefs = coefs[(coefs["model"] == "lasso") & (coefs["target"] == target)]
    if coefs.empty:
        return
    pivot = coefs.pivot(index="feature", columns="held_out_dataset",
                        values="coefficient").fillna(0.0)
    # Sort by mean absolute coefficient
    pivot["_mean_abs"] = pivot.abs().mean(axis=1)
    pivot = pivot.sort_values("_mean_abs", ascending=False).drop(columns="_mean_abs")
    pivot = pivot.head(top_k)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(pivot))))
    im = ax.imshow(pivot.values, cmap="RdBu_r",
                   vmin=-np.abs(pivot.values).max(),
                   vmax=np.abs(pivot.values).max(), aspect="auto")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    plt.colorbar(im, ax=ax, label="Lasso coefficient")
    ax.set_title(f"Lasso coefficient stability across LODO folds — {target}")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_residuals_by_dataset(b_out: dict, target: str, path: Path):
    sub = b_out["predictions"]
    sub = sub[sub["target"] == target].copy()
    sub["residual"] = sub["y_true_z"] - sub["y_pred_z"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, name in zip(axes, ["linear", "ridge", "lasso"]):
        s = sub[sub["model"] == name]
        groups = [s[s["held_out_dataset"] == d]["residual"].values
                  for d in sorted(s["held_out_dataset"].unique())]
        ax.boxplot(groups, labels=sorted(s["held_out_dataset"].unique()))
        ax.axhline(0, color="k", lw=0.8, alpha=0.5)
        ax.set_title(f"{name} — LODO residuals")
        ax.set_ylabel("y_true_z − y_pred_z")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Residuals by held-out dataset — {target}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Pipeline ─────────────────────────────────────────────────────────

def run(per_dataset_path: Path, features_path: Path, out_dir: Path,
        targets: list[str] = ("y_mean_shadow_shift", "y_var_shadow_shift")):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    df = load_data(per_dataset_path, features_path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns "
          f"({df['dataset'].nunique()} datasets × "
          f"{df['graph_id'].nunique()} unique graphs)")

    all_results = []
    all_coefs = []
    all_lodo_preds = []

    all_kfold_preds = []
    for target in targets:
        print(f"\n── Target: {target} ──")
        a = strategy_a(df, target, out_dir)
        b = strategy_b(df, target, out_dir)
        c = strategy_c(df, target, out_dir)
        d = strategy_d(df, target, out_dir)

        all_results.append(a["results"])
        all_results.append(b["results"])
        all_results.append(c["results"])
        all_results.append(d["results"])
        all_coefs.append(a["coefficients"])
        all_coefs.append(b["coefficients"])
        all_coefs.append(c["coefficients"])
        all_lodo_preds.append(b["predictions"])
        all_kfold_preds.append(c["predictions"])

        print("Strategy A (in-sample, dataset dummies):")
        print(a["results"][["model", "n_features", "n_nonzero",
                            "in_sample_R2", "in_sample_MAE"]].to_string(index=False))
        print("Strategy B (LODO CV, within-dataset z-scored):")
        print(b["results"][["model", "n_features", "lodo_R2",
                            "lodo_MAE"]].to_string(index=False))
        print("Strategy C (5-fold random CV, mixed rows, topology + dataset dummies):")
        print(c["results"][["model", "n_features", "cv_R2",
                            "cv_MAE"]].to_string(index=False))
        print("Strategy D (5-fold random CV, BASELINE: dataset + minority_ratio only):")
        print(d["results"][["model", "n_features", "cv_R2",
                            "cv_MAE"]].to_string(index=False))

        # Plots
        plot_predicted_vs_actual(a, b, target,
            out_dir / "plots" / f"predicted_vs_actual_{target}.png")
        plot_coef_stability(b, target,
            out_dir / "plots" / f"coef_stability_lasso_{target}.png")
        plot_residuals_by_dataset(b, target,
            out_dir / "plots" / f"residuals_by_dataset_{target}.png")

    results_df = pd.concat(all_results, ignore_index=True)
    coefs_df = pd.concat(all_coefs, ignore_index=True)
    lodo_df = pd.concat(all_lodo_preds, ignore_index=True)

    results_df.to_csv(out_dir / "regression_results.csv", index=False)
    coefs_df[coefs_df["strategy"] == "A_dummies"].to_csv(
        out_dir / "coefficients_strategy_A.csv", index=False)
    coefs_df[coefs_df["strategy"] == "B_lodo"].to_csv(
        out_dir / "coefficients_strategy_B.csv", index=False)
    coefs_df[coefs_df["strategy"] == "C_kfold"].to_csv(
        out_dir / "coefficients_strategy_C.csv", index=False)
    lodo_df.to_csv(out_dir / "lodo_predictions.csv", index=False)
    if all_kfold_preds:
        pd.concat(all_kfold_preds, ignore_index=True).to_csv(
            out_dir / "kfold_predictions.csv", index=False)

    print(f"\n=== Done ===")
    print(f"Outputs in: {out_dir}")
    return results_df, coefs_df, lodo_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_dataset", type=Path,
                        default=Path("outputs/graph_features/joined/per_graph_per_dataset.csv"))
    parser.add_argument("--features", type=Path, default=None,
                        help="Optional separate features CSV; not needed if "
                             "per_dataset already has features merged in.")
    parser.add_argument("--out_dir", type=Path,
                        default=Path("outputs/regression"))
    parser.add_argument("--targets", nargs="+",
                        default=["y_mean_shadow_shift", "y_var_shadow_shift"])
    args = parser.parse_args()
    run(args.per_dataset, args.features, args.out_dir, targets=args.targets)