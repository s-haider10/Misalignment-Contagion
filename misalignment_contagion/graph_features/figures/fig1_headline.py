"""Figure 1 (HEADLINE): Topology predicts persistence, not magnitude.

Split into two files:
  fig1a_headline.png  =  Panel A (trajectory schematic) + Panel B (ΔR² bars)
  fig1b_predicted_vs_actual.png  =  Panel C (predicted-vs-actual scatter)

Panel B is per-dataset only (pooled bars dropped per author request).
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .fig_style import (
    apply, panel_label, style_axes, add_zero_rule,
    DATASET_COLORS, DATASET_LABELS, FAMILY_COLORS, FAMILY_ORDER, FAMILY_LABELS,
    ROLE_COLORS, MUTED, GRID, RULE,
)
from .data import (
    all_datasets_per_agent, graph_level_from_per_agent, FIG_DIR, ensure_dir,
)
from .stats_helpers import delta_r2, cv_predict, baseline_plus_topology

apply()

OUT_1A_PNG = ensure_dir(FIG_DIR) / "fig1a_headline.png"
OUT_1A_PDF = ensure_dir(FIG_DIR) / "fig1a_headline.pdf"
OUT_1B_PNG = ensure_dir(FIG_DIR) / "fig1b_predicted_vs_actual.png"
OUT_1B_PDF = ensure_dir(FIG_DIR) / "fig1b_predicted_vs_actual.pdf"

OUTCOMES = [
    ("y_mean_shift", "Mean shift (EV)"),
    ("y_mean_II",    "Mean II"),
    ("y_var_II",     "Var II"),
    ("y_mean_SRF",   "Mean SRF"),
    ("y_var_SRF",    "Var SRF"),
]

DATASETS = ["synthetic", "moral_stories",
            "harmbench_standard", "harmbench_contextual", "harmbench_copyright"]


# ── Panel A: schematic trajectories ──────────────────────────────────
def panel_a(ax):
    rounds = np.array([0, 1, 2, 3, 4, 5, 6])  # 0=baseline, 1-5=rounds, 6=shadow
    # misaligned reference (stays high)
    mis = np.array([6.5, 6.5, 6.5, 6.6, 6.5, 6.5, 6.4])
    # Compliance: shifts during deliberation, reverts privately
    comp = np.array([4.0, 4.5, 5.2, 5.8, 6.0, 6.1, 4.2])
    # Internalization: shifts and stays
    inter = np.array([4.0, 4.4, 4.9, 5.3, 5.5, 5.7, 5.5])

    # Misaligned reference
    ax.plot(rounds, mis, color=ROLE_COLORS["misaligned"], linewidth=1.6,
            linestyle="--", alpha=0.85, label="Misaligned ref.")
    # Compliance (low II)
    ax.plot(rounds, comp, color="#79b8ff", linewidth=2.4,
            marker="o", markersize=5, label="Compliance (II ≈ 0)")
    # Internalization (high II)
    ax.plot(rounds, inter, color="#0a3d8f", linewidth=2.4,
            marker="s", markersize=5, label="Internalization (II ≈ 1)")

    # Stage shading
    ax.axvspan(-0.4, 0.4, color="#f5f5f5", alpha=0.6, zorder=0)
    ax.axvspan(0.4, 5.4, color="#fbfbf0", alpha=0.5, zorder=0)
    ax.axvspan(5.4, 6.4, color="#f5f5f5", alpha=0.6, zorder=0)

    # Stage labels
    ax.text(0.0, 1.5, "Baseline", ha="center", fontsize=8, color=MUTED, style="italic")
    ax.text(3.0, 1.5, "Deliberation (5 rounds)", ha="center", fontsize=8, color=MUTED, style="italic")
    ax.text(6.0, 1.5, "Shadow", ha="center", fontsize=8, color=MUTED, style="italic")

    # Brackets for the two outcome interpretations
    # Mean shift is measured at the SHADOW stage, not during deliberation:
    # mean shift = shadow EV − baseline EV (per aligned agent, then averaged).
    ax.annotate("", xy=(6, 4.25), xytext=(6, 6.0),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.8))
    ax.text(6.18, 5.1, "Mean shift\n(shadow EV −\nbaseline EV)",
            fontsize=7.5, color=MUTED, va="center")
    # II = JSD(shadow,baseline) / JSD(final,baseline): high if shadow ≈ final
    ax.text(5.5, 3.05,
            "II → 1 if shadow ≈ final\n(internalization)\n"
            "II → 0 if shadow ≈ baseline\n(Asch compliance)",
            fontsize=7.5, color=MUTED, va="top", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=GRID))

    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_xticklabels(["B", "R1", "R2", "R3", "R4", "R5", "S"])
    ax.set_xlim(-0.5, 7.0)
    ax.set_ylim(1, 7.5)
    ax.set_yticks([1, 3, 5, 7])
    ax.set_xlabel("Stage")
    ax.set_ylabel("Stance EV (1–7)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=8)
    style_axes(ax, grid_axis="y")


# ── Panel B: ΔR² bars across outcomes  ──────────────────────────────
def compute_delta_r2_pooled(gl_by_dataset, datasets, outcomes):
    """Pooled across datasets with `dataset` as a covariate in the BASELINE.

    Tests whether topology adds predictive power over (ratio + n_agents +
    n_misaligned + dataset_identity). This is the analytically honest baseline
    because we want to know whether graph structure carries information that
    isn't already captured by composition + content domain.
    """
    from .stats_helpers import CONTROL_COLS, topology_cols, cv_r2

    pooled = pd.concat([gl_by_dataset[ds] for ds in datasets if ds in gl_by_dataset],
                       ignore_index=True)
    topo = topology_cols(pooled)
    dummies = pd.get_dummies(pooled["dataset"], prefix="ds",
                              drop_first=True).astype(float).values

    rows = []
    for target, label in outcomes:
        y = pooled[target].values
        X_base = np.hstack([pooled[CONTROL_COLS].values, dummies])
        X_full = np.hstack([X_base, pooled[topo].values])
        r2_b = cv_r2(X_base, y, "lasso")
        r2_f = cv_r2(X_full, y, "lasso")
        rows.append({"target": target, "label": label,
                     "r2_baseline": r2_b, "r2_full": r2_f,
                     "delta": r2_f - r2_b})
    return pd.DataFrame(rows)


def compute_delta_r2_per_dataset(gl_by_dataset, datasets, outcomes):
    """Per-dataset (no dataset dummy) — supplementary breakdown shown as a
    secondary set of bars on the same axes."""
    rows = []
    for ds in datasets:
        if ds not in gl_by_dataset:
            continue
        gl = gl_by_dataset[ds]
        for target, label in outcomes:
            res = delta_r2(gl, target, model="lasso")
            rows.append({"dataset": ds, "target": target, "label": label,
                         "delta": res["r2_full"] - res["r2_baseline"]})
    return pd.DataFrame(rows)


def panel_b(ax, gl_by_dataset, datasets, outcomes):
    """Per-dataset ΔR² only. Pooled bar dropped per author request."""
    per_ds_df = compute_delta_r2_per_dataset(gl_by_dataset, datasets, outcomes)

    outcome_labels = [lbl for _, lbl in outcomes]
    n_o = len(outcomes)
    x = np.arange(n_o)

    series = [
        ("synthetic",            "Synthetic",          DATASET_COLORS["synthetic"]),
        ("moral_stories",        "Moral Stories",      DATASET_COLORS["moral_stories"]),
        ("harmbench_standard",   "HarmBench (std)",    DATASET_COLORS["harmbench_standard"]),
        ("harmbench_contextual", "HarmBench (ctx)",    DATASET_COLORS["harmbench_contextual"]),
        ("harmbench_copyright",  "HarmBench (cpy)",    DATASET_COLORS["harmbench_copyright"]),
    ]
    n_b = len(series)
    width = 0.78 / n_b

    for j, (key, label, color) in enumerate(series):
        vals = []
        for tgt, _ in outcomes:
            v = per_ds_df.loc[(per_ds_df["dataset"] == key) &
                                (per_ds_df["target"] == tgt), "delta"].values
            vals.append(float(v[0]) if len(v) else 0.0)
        pos = x + (j - (n_b - 1) / 2) * width
        ax.bar(pos, vals, width=width, color=color, label=label,
               edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(outcome_labels, fontsize=9)
    ax.set_ylabel(r"$\Delta R^2$  (topology added over baseline)")
    add_zero_rule(ax, axis="y")
    style_axes(ax, grid_axis="y")
    ax.legend(loc="upper right", fontsize=8.5, ncol=1)


# ── Panel C: predicted vs actual scatter for mean II ────────────────
def panel_c(axes, gl_by_dataset, datasets):
    """Two side-by-side scatters: synthetic, moral_stories.
    Returns the axes (for figure-level legend extraction)."""
    targets_for_scatter = [d for d in datasets if d in gl_by_dataset and d != "harmbench_standard"]
    for ax, ds in zip(axes, targets_for_scatter):
        gl = gl_by_dataset[ds]
        y = gl["y_mean_II"].values
        X = baseline_plus_topology(gl)
        y_true, y_pred = cv_predict(X, y, model="lasso")
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)

        # Color by family
        for fam in FAMILY_ORDER:
            mask = gl["family"] == fam
            if mask.sum() == 0:
                continue
            ax.scatter(y_true[mask], y_pred[mask],
                       color=FAMILY_COLORS[fam], s=18, alpha=0.78,
                       edgecolor="white", linewidth=0.4,
                       label=FAMILY_LABELS[fam], zorder=3)
        # Diagonal
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], color=RULE, linewidth=0.9,
                linestyle="--", zorder=1)
        ax.set_xlabel("Actual mean II")
        ax.set_ylabel("Predicted mean II")
        ax.set_title(DATASET_LABELS[ds], fontsize=10, pad=6)
        ax.text(0.05, 0.93, rf"$R^2 = {r2:.3f}$", transform=ax.transAxes,
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor=GRID))
        style_axes(ax, grid_axis="both")

    return axes  # caller will pull legend handles for figure-level placement


def build():
    pa = all_datasets_per_agent()
    gl = {ds: graph_level_from_per_agent(df, ds) for ds, df in pa.items()}

    # ── Figure 1a: Panel A + Panel B ──────────────────────────────────
    fig1a = plt.figure(figsize=(13, 4.4))
    gs1a = gridspec.GridSpec(1, 2, figure=fig1a, wspace=0.32,
                              left=0.07, right=0.97, top=0.92, bottom=0.18)
    ax_a = fig1a.add_subplot(gs1a[0, 0])
    ax_b = fig1a.add_subplot(gs1a[0, 1])
    panel_a(ax_a)
    panel_b(ax_b, gl, DATASETS, OUTCOMES)
    panel_label(ax_a, "A")
    panel_label(ax_b, "B")
    fig1a.savefig(OUT_1A_PNG, facecolor="white", bbox_inches="tight")
    fig1a.savefig(OUT_1A_PDF, facecolor="white", bbox_inches="tight")
    plt.close(fig1a)
    print(f"wrote {OUT_1A_PNG.relative_to(OUT_1A_PNG.parents[2])}")
    print(f"wrote {OUT_1A_PDF.relative_to(OUT_1A_PDF.parents[2])}")

    # ── Figure 1b: Panel C standalone (predicted vs actual) ──────────
    fig1b = plt.figure(figsize=(11.5, 4.6))
    gs1b = gridspec.GridSpec(1, 2, figure=fig1b, wspace=0.28,
                              left=0.07, right=0.97, top=0.92, bottom=0.20)
    ax_c1 = fig1b.add_subplot(gs1b[0, 0])
    ax_c2 = fig1b.add_subplot(gs1b[0, 1])
    c_axes = panel_c([ax_c1, ax_c2], gl, DATASETS)
    handles, labels = c_axes[-1].get_legend_handles_labels()
    fig1b.legend(handles, labels, loc="lower center",
                  bbox_to_anchor=(0.5, -0.005), ncol=7, fontsize=8.5,
                  columnspacing=1.2, handletextpad=0.4, frameon=False)
    fig1b.savefig(OUT_1B_PNG, facecolor="white", bbox_inches="tight")
    fig1b.savefig(OUT_1B_PDF, facecolor="white", bbox_inches="tight")
    plt.close(fig1b)
    print(f"wrote {OUT_1B_PNG.relative_to(OUT_1B_PNG.parents[2])}")
    print(f"wrote {OUT_1B_PDF.relative_to(OUT_1B_PDF.parents[2])}")


if __name__ == "__main__":
    build()
