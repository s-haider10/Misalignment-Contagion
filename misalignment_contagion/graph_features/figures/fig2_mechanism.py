"""Figure 2: Mechanism — per-agent topological features predict individual-level persistence.

Panel A: Standardized coefficient heatmap (features × dataset-outcome cells)
Panel B: Within-graph FE R² decomposition (graph_id alone vs + agent features)
Panel C: Marginal effect of in_degree on II
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from .fig_style import (
    apply, panel_label, style_axes, add_zero_rule,
    DATASET_COLORS, DATASET_LABELS, MUTED, GRID, RULE, DIVERGING_CMAP,
)
from .data import all_datasets_per_agent, FIG_DIR, ensure_dir
from .stats_helpers import AGENT_FEATURE_COLS

warnings.filterwarnings("ignore")
apply()

OUT_PATH = ensure_dir(FIG_DIR) / "fig2_mechanism.png"
OUT_PDF = ensure_dir(FIG_DIR) / "fig2_mechanism.pdf"

DATASETS = ["synthetic", "moral_stories"]  # harmbench excluded (scope condition; see Fig 3)
TARGETS = ["II", "SRF"]

FEATURE_LABELS = {
    "in_degree":                  "In-degree",
    "out_degree":                 "Out-degree",
    "n_neighbors":                "Neighbors",
    "frac_neighbors_misaligned":  "Frac. neighbors misaligned",
    "n_misaligned_1hop":          "Misaligned 1-hop",
    "n_misaligned_2hop":          "Misaligned 2-hop",
    "dist_to_nearest_misaligned": "Distance to nearest misaligned",
    "betweenness":                "Betweenness",
    "closeness":                  "Closeness",
}
FEATURE_ORDER = [
    "in_degree", "out_degree", "n_neighbors",
    "frac_neighbors_misaligned", "n_misaligned_1hop", "n_misaligned_2hop",
    "dist_to_nearest_misaligned", "betweenness", "closeness",
]


def get_standardized_coefs(df_ag, target):
    """Standardized OLS coefs for the 9 agent features predicting target."""
    sub = df_ag.dropna(subset=[target] + AGENT_FEATURE_COLS).reset_index(drop=True)
    X = StandardScaler().fit_transform(sub[AGENT_FEATURE_COLS].values)
    # Normalize y as well so coefficients are comparable across datasets/outcomes
    y_std = (sub[target].values - sub[target].mean()) / sub[target].std()
    m = LinearRegression().fit(X, y_std)
    return dict(zip(AGENT_FEATURE_COLS, m.coef_))


# ── Panel A: coefficient heatmap ─────────────────────────────────────
def panel_a(ax, pa_by_dataset):
    """Coefficient heatmap. Columns ordered as (II × {Syn, MS}) | (SRF × {Syn, MS})
    so that within-outcome cross-dataset comparison is one saccade left↔right."""
    cells = []
    col_labels = []
    for tgt in TARGETS:
        for ds in DATASETS:
            if ds not in pa_by_dataset:
                continue
            cells.append(get_standardized_coefs(pa_by_dataset[ds], tgt))
            # Single-line label, more legible when rotated than stacked 2-line
            col_labels.append(f"{tgt} · {DATASET_LABELS[ds]}")

    M = np.zeros((len(FEATURE_ORDER), len(cells)))
    for j, c in enumerate(cells):
        for i, f in enumerate(FEATURE_ORDER):
            M[i, j] = c.get(f, np.nan)
    vmax = np.nanmax(np.abs(M))
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    im = ax.imshow(M, cmap=DIVERGING_CMAP, norm=norm, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8.5, rotation=30, ha="right",
                        rotation_mode="anchor")
    ax.set_yticks(range(len(FEATURE_ORDER)))
    ax.set_yticklabels([FEATURE_LABELS[f] for f in FEATURE_ORDER], fontsize=8.5)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt_color = "white" if abs(v) > 0.55 * vmax else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=7.5, color=txt_color)

    # Separator between outcome blocks (II vs SRF)
    n_ds = len(DATASETS)
    for k in range(1, len(TARGETS)):
        ax.axvline(k * n_ds - 0.5, color="white", linewidth=2.6)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.04)
    cbar.set_label("Std. OLS coefficient (y also z-scored)", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("Per-agent coefficients (signs replicate across datasets)",
                 fontsize=10, pad=8)


# ── Panel B: within-graph FE R² decomposition ────────────────────────
def compute_fe_decomposition(df_ag, target):
    sub = df_ag.dropna(subset=[target] + AGENT_FEATURE_COLS).reset_index(drop=True)
    y = sub[target].values
    n = len(sub)
    gid_codes = pd.Categorical(sub["graph_id"]).codes
    n_u = gid_codes.max() + 1
    fe = sp.csr_matrix((np.ones(n), (np.arange(n), gid_codes)),
                       shape=(n, n_u))[:, 1:]
    agent_std = StandardScaler().fit_transform(sub[AGENT_FEATURE_COLS].values)

    def cv_sp(X):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        yt, yp = [], []
        for tr, te in kf.split(np.arange(n)):
            m = Ridge(alpha=10.0, solver="lsqr"); m.fit(X[tr], y[tr])
            yt.append(y[te]); yp.append(m.predict(X[te]))
        return r2_score(np.concatenate(yt), np.concatenate(yp))

    r2_fe_only = max(0.0, cv_sp(fe))
    r2_fe_plus = max(0.0, cv_sp(sp.hstack([fe, sp.csr_matrix(agent_std)], format="csr")))
    return {
        "fe_only": r2_fe_only,
        "agent_add": max(0.0, r2_fe_plus - r2_fe_only),
        "residual": max(0.0, 1.0 - r2_fe_plus),
    }


def panel_b(ax, pa_by_dataset):
    """Single ΔR² bar chart: how much do agent features add ON TOP of graph_id
    fixed effects? Each bar is a (outcome × dataset) cell."""
    labels = []
    deltas = []
    bar_colors = []
    for tgt in TARGETS:
        for ds in DATASETS:
            if ds not in pa_by_dataset:
                continue
            d = compute_fe_decomposition(pa_by_dataset[ds], tgt)
            labels.append(f"{tgt} · {DATASET_LABELS[ds]}")
            deltas.append(d["agent_add"])
            bar_colors.append(DATASET_COLORS[ds])

    x = np.arange(len(labels))
    width = 0.62
    ax.bar(x, deltas, width=width, color=bar_colors,
           edgecolor="white", linewidth=0.6, zorder=3)
    for i, v in enumerate(deltas):
        ax.text(i, v + 0.001, f"+{v:.4f}", ha="center", va="bottom",
                fontsize=8, color="#0a3d8f", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, rotation=30, ha="right",
                        rotation_mode="anchor")
    ax.set_ylabel(r"$\Delta R^2$  added by agent features")
    ymax = max(deltas) * 1.4 if deltas else 0.05
    ax.set_ylim(0, max(0.05, ymax))
    style_axes(ax, grid_axis="y")
    ax.set_title("Within-graph agent contribution\n(over graph-id fixed effects)",
                 fontsize=10, pad=8)
    ax.text(0.02, 0.97,
            "Effect is small but consistent across\n"
            "datasets and outcomes; agent position\n"
            "matters even within a fixed graph.",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, color=MUTED, style="italic")


# ── Panel C: marginal effect of in_degree on II ─────────────────────
def panel_c(ax, pa_by_dataset):
    for ds in DATASETS:
        if ds not in pa_by_dataset:
            continue
        sub = pa_by_dataset[ds].dropna(subset=["II"] + AGENT_FEATURE_COLS).copy()
        # Fit OLS with all controls + agent features (no FE so marginal interpretable)
        X = StandardScaler().fit_transform(sub[AGENT_FEATURE_COLS].values)
        y = sub["II"].values
        m = LinearRegression().fit(X, y)
        # Vary in_degree (col 5 = "in_degree"), holding others at mean (0 after std)
        in_idx = AGENT_FEATURE_COLS.index("in_degree")
        zs = np.linspace(-2, 2, 50)
        X_grid = np.zeros((len(zs), len(AGENT_FEATURE_COLS)))
        X_grid[:, in_idx] = zs
        y_pred = m.predict(X_grid)

        # Bootstrap CI
        rng = np.random.default_rng(42)
        n_boot = 80
        sample_n = min(20000, len(sub))
        boot_preds = np.zeros((n_boot, len(zs)))
        for b in range(n_boot):
            idx = rng.choice(len(sub), size=sample_n, replace=True)
            Xs = StandardScaler().fit_transform(sub[AGENT_FEATURE_COLS].values[idx])
            mb = LinearRegression().fit(Xs, y[idx])
            boot_preds[b] = mb.predict(X_grid)
        lo, hi = np.percentile(boot_preds, [2.5, 97.5], axis=0)

        # II is a non-negative ratio; clip predictions and CI band at 0
        y_pred_c = np.clip(y_pred, 0, None)
        lo_c = np.clip(lo, 0, None)
        hi_c = np.clip(hi, 0, None)
        color = DATASET_COLORS[ds]
        ax.plot(zs, y_pred_c, color=color, linewidth=2.0,
                label=DATASET_LABELS[ds], zorder=3)
        ax.fill_between(zs, lo_c, hi_c, color=color, alpha=0.18, zorder=2)

    ax.set_xlabel("In-degree (z-score)")
    ax.set_ylabel("Predicted II (≥ 0)")
    ax.set_ylim(bottom=0)
    ax.set_title("Marginal effect of in-degree on II", fontsize=10, pad=8)
    add_zero_rule(ax, axis="x")
    ax.legend(loc="upper right", fontsize=8.5)
    style_axes(ax, grid_axis="both")
    ax.text(0.04, 0.04,
            "More visibility → less\nprivate internalization\n(prediction clipped at 0)",
            transform=ax.transAxes, fontsize=8, color=MUTED, style="italic",
            va="bottom")


def build():
    pa = all_datasets_per_agent()

    fig = plt.figure(figsize=(14, 5.8))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.35, 1.0, 0.95],
                            wspace=0.55)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_a(ax_a, pa)
    panel_b(ax_b, pa)
    panel_c(ax_c, pa)

    panel_label(ax_a, "A", x=-0.32)
    panel_label(ax_b, "B")
    panel_label(ax_c, "C")

    fig.savefig(OUT_PATH, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_PATH.relative_to(OUT_PATH.parents[2])}")


if __name__ == "__main__":
    build()
