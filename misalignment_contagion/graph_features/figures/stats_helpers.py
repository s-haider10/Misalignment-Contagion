"""Regression / R² helpers shared across the figure scripts.

All CV-based and tied to a fixed seed for reproducibility.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

CONTROL_COLS = ["minority_ratio", "n_agents", "n_misaligned"]
AGENT_FEATURE_COLS = [
    "dist_to_nearest_misaligned", "n_misaligned_1hop", "n_misaligned_2hop",
    "n_neighbors", "frac_neighbors_misaligned",
    "in_degree", "out_degree", "closeness", "betweenness",
]


def topology_cols(df: pd.DataFrame) -> list[str]:
    drop = {"graph_id", "family", "dataset", "placement_summary",
            "n_obs", "n_trials", "n_aligned_obs",
            "y_mean_shift", "y_var_shift",
            "y_mean_II", "y_var_II", "y_mean_SRF", "y_var_SRF"}
    return [c for c in df.columns
            if c not in drop and c not in CONTROL_COLS
            and np.issubdtype(df[c].dtype, np.number)]


def cv_r2(X: np.ndarray, y: np.ndarray, model: str = "lasso",
          n_splits: int = 5, seed: int = 42) -> float:
    """5-fold CV R² with the requested model."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yt_all, yp_all = [], []
    for tr, te in kf.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        if model == "lasso":
            m = LassoCV(alphas=np.logspace(-3, 1, 50), cv=3,
                        max_iter=20000, random_state=42)
        elif model == "ridge":
            m = RidgeCV(alphas=np.logspace(-3, 3, 25), cv=3)
        else:
            m = LinearRegression()
        m.fit(Xtr, y[tr])
        yt_all.append(y[te])
        yp_all.append(m.predict(Xte))
    return r2_score(np.concatenate(yt_all), np.concatenate(yp_all))


def cv_predict(X: np.ndarray, y: np.ndarray, model: str = "lasso",
               n_splits: int = 5, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return out-of-fold (y_true, y_pred) arrays aligned to input row order."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred = np.empty_like(y, dtype=float)
    for tr, te in kf.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        if model == "lasso":
            m = LassoCV(alphas=np.logspace(-3, 1, 50), cv=3,
                        max_iter=20000, random_state=42)
        elif model == "ridge":
            m = RidgeCV(alphas=np.logspace(-3, 3, 25), cv=3)
        else:
            m = LinearRegression()
        m.fit(Xtr, y[tr])
        y_pred[te] = m.predict(Xte)
    return y, y_pred


def baseline_design(df: pd.DataFrame) -> np.ndarray:
    """minority_ratio + n_agents + n_misaligned."""
    return df[CONTROL_COLS].values


def baseline_plus_topology(df: pd.DataFrame) -> np.ndarray:
    """controls + every topology feature."""
    topo = topology_cols(df)
    return np.hstack([df[CONTROL_COLS].values, df[topo].values])


def delta_r2(df: pd.DataFrame, target: str,
             model: str = "lasso", seed: int = 42) -> dict:
    """Return baseline R², full R², Δ, and per-fold splits for one outcome."""
    y = df[target].values
    X_base = baseline_design(df)
    X_full = baseline_plus_topology(df)
    return {
        "target": target,
        "r2_baseline": cv_r2(X_base, y, model=model, seed=seed),
        "r2_full":     cv_r2(X_full, y, model=model, seed=seed),
        "n_rows": len(df),
    }


def standardized_ols_coefs(df: pd.DataFrame, target: str,
                           feature_cols: list[str]) -> pd.DataFrame:
    sub = df.dropna(subset=[target] + feature_cols).reset_index(drop=True)
    X = StandardScaler().fit_transform(sub[feature_cols].values)
    y = sub[target].values
    m = LinearRegression().fit(X, y)
    return pd.DataFrame({"feature": feature_cols, "coef": m.coef_})
