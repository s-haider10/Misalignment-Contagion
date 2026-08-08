"""Figure 8: Scope-condition heatmap across (dataset × outcome).

For each dataset × outcome cell, show ΔR² (topology added over baseline).
Diverging colormap centered on 0 makes the dissociation visible at a glance:
where topology helps (positive) vs hurts (negative) vs adds nothing (~0).
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from .fig_style import (apply, style_axes, DATASET_LABELS, MUTED, GRID,
                          DIVERGING_CMAP)
from .data import FIG_DIR, ensure_dir

warnings.filterwarnings("ignore")
apply()

OUT_PATH = ensure_dir(FIG_DIR) / "fig8_scope_heatmap.png"
OUT_PDF = ensure_dir(FIG_DIR) / "fig8_scope_heatmap.pdf"

DATASET_ORDER = ["synthetic", "moral_stories",
                 "harmbench_copyright", "harmbench_standard",
                 "harmbench_contextual"]
OUTCOMES = [
    ("y_mean_shift",    "Mean shift\n(EV)"),
    ("y_mean_abs_shift","Mean |shift|\n(EV)"),
    ("y_mean_II",       "Mean II\n(persistence)"),
    ("y_var_II",        "Var II"),
    ("y_mean_SRF",      "Mean SRF"),
    ("y_var_SRF",       "Var SRF"),
]


def build():
    df = pd.read_csv("outputs/analysis/regression_summary.csv")
    # Use Lasso (sparse, less noisy than GBM for this number of graphs)
    df = df[(df["model"] == "lasso") & (df["dataset"] != "POOLED_with_dummy")]

    targets = [t for t, _ in OUTCOMES]
    M = np.full((len(DATASET_ORDER), len(targets)), np.nan)
    for i, ds in enumerate(DATASET_ORDER):
        for j, t in enumerate(targets):
            v = df[(df["dataset"] == ds) & (df["target"] == t)]["delta_r2_topology"]
            if len(v) == 1:
                M[i, j] = float(v.iloc[0])

    vmax = float(np.nanmax(np.abs(M)))
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(M, cmap=DIVERGING_CMAP, norm=norm, aspect="auto")

    ax.set_xticks(range(len(OUTCOMES)))
    ax.set_xticklabels([lbl for _, lbl in OUTCOMES], fontsize=9)
    ax.set_yticks(range(len(DATASET_ORDER)))
    ax.set_yticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=9)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            color = "white" if abs(v) > 0.55 * vmax else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=8.5, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label(r"$\Delta R^2$  (topology added over baseline, Lasso)",
                    fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Horizontal separator between normative datasets and harmbench
    ax.axhline(1.5, color="white", linewidth=2.4)

    ax.set_title("Where topology predicts contagion: dataset × outcome",
                 fontsize=11, fontweight="bold", pad=10)

    # Side labels on the LEFT (no clipping)
    ax.text(-0.27, 0.83, "Normative",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="center",
            ha="right", style="italic", fontweight="bold")
    ax.text(-0.27, 0.33, "Harmful\n(scope cond.)",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="center",
            ha="right", style="italic", fontweight="bold")

    fig.savefig(OUT_PATH, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT_PDF, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PATH.relative_to(OUT_PATH.parents[2])}")


if __name__ == "__main__":
    build()
