"""
plots.py — Publication-quality figures for the Misalignment Contagion experiment.

Figures:
  1. Core Nature figure: 2x4 belief trajectory + internalization scatter
  2. Internalization Index heatmap (topology x ratio)
  3. Entropy crystallization trajectories
  4. Minority ratio dose-response
  5. Cross-dataset / model generalization
  6. Star topology cascade (hub vs leaf)
  7. Prompt-induced vs model-induced equivalence
  8. Prompt rigidity 2x2 interaction

Usage:
  python -m misalignment_contagion.plots --experiment primary
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns

from .config import OUTPUTS_DIR, N_ROUNDS
from .analyze import load_trials, trials_to_dataframe, CONDITION_COLS

warnings.filterwarnings("ignore")

# ── style constants ───────────────────────────────────────────────────────────
TOPO_COLORS = {
    "fc":     "#4C72B0",
    "circle": "#DD8452",
    "star":   "#55A868",
    "chain":  "#C44E52",
}
TOPO_LABELS = {"fc": "FC", "circle": "Circle", "star": "Star", "chain": "Chain"}
RATIO_COLORS = {0.1: "#6BAED6", 0.2: "#2171B5", 0.3: "#08306B"}
RATIO_LABELS = {0.1: "10%", 0.2: "20%", 0.3: "30%"}
TOPO_ORDER = ["fc", "circle", "star", "chain"]
STAGE_LABELS = ["Baseline"] + [f"R{i}" for i in range(N_ROUNDS)] + ["Shadow"]

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
})


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(fig, name: str, out_dir: str):
    path = os.path.join(out_dir, name + ".png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


def _get_ev_trajectory(row: pd.Series) -> list[float] | None:
    """Extract [baseline_ev, r0_ev, ..., r4_ev, shadow_ev] from a row."""
    if row.get("baseline_ev") is None or row.get("shadow_ev") is None:
        return None
    round_evs = row.get("round_evs", [])
    if not round_evs or any(v is None for v in round_evs):
        return None
    return [row["baseline_ev"]] + list(round_evs) + [row["shadow_ev"]]


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Core Nature Figure: 2x4 trajectory + internalization scatter
# ══════════════════════════════════════════════════════════════════════════════

def fig1_core_nature(df: pd.DataFrame, out_dir: str, ratio: float = 0.2):
    """The central figure: trajectory (row 1) + public-vs-private scatter (row 2)."""
    print("Generating fig1_core_nature ...")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor("white")
    x = np.arange(len(STAGE_LABELS))

    for ci, topo in enumerate(TOPO_ORDER):
        sub = df[(df["topology"] == topo) & (df["minority_ratio"] == ratio) &
                 (df["position_config"] == 0)]

        # ── Row 1: EV Trajectory ──
        ax = axes[0, ci]
        trajectories = []
        for _, row in sub.iterrows():
            traj = _get_ev_trajectory(row)
            if traj:
                trajectories.append(traj)

        if trajectories:
            traj_arr = np.array(trajectories)
            means = traj_arr.mean(axis=0)
            stds = traj_arr.std(axis=0, ddof=1)
            n = len(traj_arr)
            ci_band = 1.96 * stds / math.sqrt(n)

            ax.plot(x, means, "o-", color=TOPO_COLORS[topo], linewidth=2,
                    markersize=5, zorder=3)
            ax.fill_between(x, means - ci_band, means + ci_band,
                            color=TOPO_COLORS[topo], alpha=0.15, zorder=2)

            # Shadow is after a gap — draw a visual separator
            ax.axvline(x[-2] + 0.5, color="#CCCCCC", linestyle=":", linewidth=0.8)

        ax.axhline(4, color="#999999", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(STAGE_LABELS, fontsize=7, rotation=45, ha="right")
        ax.set_ylim(1, 7)
        ax.set_title(TOPO_LABELS[topo], fontsize=12, fontweight="bold")
        if ci == 0:
            ax.set_ylabel("Mean Logprob EV", fontsize=10)
        ax.spines["bottom"].set_color("#AAAAAA")
        ax.spines["left"].set_color("#AAAAAA")

        # Annotate SRF and FDR
        srf_vals = sub["shadow_reversion_fraction"].dropna()
        fdr_vals = sub["first_round_dominance"].dropna()
        ann_parts = []
        if len(srf_vals) > 0:
            ann_parts.append(f"SRF={srf_vals.median():.2f}")
        if len(fdr_vals) > 0:
            ann_parts.append(f"FDR={fdr_vals.median():.2f}")
        if ann_parts:
            ax.text(0.02, 0.97, "\n".join(ann_parts), transform=ax.transAxes,
                    fontsize=7, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#CCCCCC", alpha=0.9))

        # ── Row 2: Public vs Private shift scatter ──
        ax2 = axes[1, ci]
        matched = sub[["baseline_ev", "final_ev", "shadow_ev", "minority_ratio"]].dropna()
        if len(matched) > 0:
            public_shift = matched["final_ev"].values - matched["baseline_ev"].values
            private_shift = matched["shadow_ev"].values - matched["baseline_ev"].values

            # Color by minority ratio
            ax2.scatter(public_shift, private_shift, s=10, alpha=0.4,
                        color=TOPO_COLORS[topo], edgecolors="none", zorder=3)

            # Diagonal: full internalization
            lims = [min(public_shift.min(), private_shift.min()) - 0.3,
                    max(public_shift.max(), private_shift.max()) + 0.3]
            ax2.plot(lims, lims, "--", color="#999999", linewidth=0.9,
                     label="Full internalization")
            ax2.axhline(0, color="#CCCCCC", linewidth=0.6)
            ax2.axvline(0, color="#CCCCCC", linewidth=0.6)

            # Fraction labels
            n_total = len(public_shift)
            n_above = int(np.sum(private_shift > public_shift))
            n_below = n_total - n_above
            ax2.text(0.98, 0.02, f"Asch: {n_below/n_total:.0%}",
                     transform=ax2.transAxes, fontsize=7, ha="right", va="bottom",
                     color="#666666")
            ax2.text(0.02, 0.98, f"Moscovici: {n_above/n_total:.0%}",
                     transform=ax2.transAxes, fontsize=7, ha="left", va="top",
                     color="#666666")

        ax2.set_xlabel("Public shift (final - baseline)", fontsize=9)
        if ci == 0:
            ax2.set_ylabel("Private shift (shadow - baseline)", fontsize=9)
        ax2.spines["bottom"].set_color("#AAAAAA")
        ax2.spines["left"].set_color("#AAAAAA")
        ax2.set_aspect("equal", adjustable="datalim")
        ax2.grid(True, color="#EEEEEE", linewidth=0.5)

    fig.suptitle(f"Belief Contagion: Trajectory and Internalization by Topology "
                 f"(minority ratio {ratio:.0%})",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig1_core_nature", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Internalization Index heatmap (topology x ratio)
# ══════════════════════════════════════════════════════════════════════════════

def fig2_ii_heatmap(df: pd.DataFrame, out_dir: str):
    print("Generating fig2_ii_heatmap ...")

    ratios = [0.1, 0.2, 0.3]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("white")

    metrics = [("internalization_index", "Internalization Index (II)"),
               ("shadow_reversion_fraction", "Shadow Reversion Fraction (SRF)")]

    for ax_idx, (col, title) in enumerate(metrics):
        ax = axes[ax_idx]
        matrix = np.full((len(TOPO_ORDER), len(ratios)), np.nan)
        counts = np.zeros_like(matrix)

        for ti, topo in enumerate(TOPO_ORDER):
            for ri, ratio in enumerate(ratios):
                sub = df[(df["topology"] == topo) &
                         (df["minority_ratio"] == ratio) &
                         (df["position_config"] == 0)]
                vals = sub[col].dropna().values
                if len(vals) > 0:
                    matrix[ti, ri] = np.median(vals)
                    counts[ti, ri] = len(vals)

        # Diverging colormap centred at the meaningful threshold
        center = 1.0 if col == "internalization_index" else 0.5
        vmin = max(0, np.nanmin(matrix) - 0.1) if not np.all(np.isnan(matrix)) else 0
        vmax = np.nanmax(matrix) + 0.1 if not np.all(np.isnan(matrix)) else 2
        cmap = "RdYlBu_r" if col == "internalization_index" else "RdYlBu"

        im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest")

        for ti in range(len(TOPO_ORDER)):
            for ri in range(len(ratios)):
                val = matrix[ti, ri]
                if np.isnan(val):
                    continue
                txt_color = "white" if abs(val - (vmin + vmax) / 2) > (vmax - vmin) * 0.3 else "black"
                ax.text(ri, ti, f"{val:.2f}\nn={int(counts[ti, ri])}",
                        ha="center", va="center", fontsize=8, color=txt_color)

        ax.set_xticks(range(len(ratios)))
        ax.set_xticklabels([RATIO_LABELS[r] for r in ratios])
        ax.set_yticks(range(len(TOPO_ORDER)))
        ax.set_yticklabels([TOPO_LABELS[t] for t in TOPO_ORDER])
        ax.set_xlabel("Minority Ratio")
        ax.set_title(title, fontsize=10)
        ax.grid(False)
        ax.spines[:].set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    fig.tight_layout()
    _save(fig, "fig2_ii_heatmap", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Entropy crystallization trajectories
# ══════════════════════════════════════════════════════════════════════════════

def fig3_entropy_trajectories(df: pd.DataFrame, out_dir: str, ratio: float = 0.2):
    print("Generating fig3_entropy_trajectories ...")

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    x = np.arange(len(STAGE_LABELS))

    for topo in TOPO_ORDER:
        sub = df[(df["topology"] == topo) & (df["minority_ratio"] == ratio) &
                 (df["position_config"] == 0)]
        trajs = sub["entropy_trajectory"].dropna().tolist()
        if not trajs:
            continue

        # Filter to full-length trajectories
        n_stages = len(STAGE_LABELS)
        valid = [t for t in trajs if len(t) == n_stages and all(v is not None for v in t)]
        if not valid:
            continue

        arr = np.array(valid)
        means = arr.mean(axis=0)
        stds = arr.std(axis=0, ddof=1)
        n = len(arr)
        ci_band = 1.96 * stds / math.sqrt(n)

        ax.plot(x, means, "o-", color=TOPO_COLORS[topo], linewidth=2,
                markersize=5, label=TOPO_LABELS[topo], zorder=3)
        ax.fill_between(x, means - ci_band, means + ci_band,
                        color=TOPO_COLORS[topo], alpha=0.12)

    ax.axvline(x[-2] + 0.5, color="#CCCCCC", linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS, fontsize=8)
    ax.set_ylabel("Shannon Entropy (bits)", fontsize=10)
    ax.set_xlabel("Stage", fontsize=10)
    ax.set_title("Belief Crystallization: Entropy Trajectory by Topology", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.spines["left"].set_color("#AAAAAA")

    fig.tight_layout()
    _save(fig, "fig3_entropy_trajectories", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Minority ratio dose-response
# ══════════════════════════════════════════════════════════════════════════════

def fig4_dose_response(df: pd.DataFrame, out_dir: str):
    print("Generating fig4_dose_response ...")

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")

    ratios = [0.1, 0.2, 0.3]

    for topo in TOPO_ORDER:
        means, cis = [], []
        for ratio in ratios:
            sub = df[(df["topology"] == topo) & (df["minority_ratio"] == ratio) &
                     (df["position_config"] == 0)]
            matched = sub[["baseline_ev", "shadow_ev"]].dropna()
            if len(matched) > 0:
                shifts = matched["shadow_ev"].values - matched["baseline_ev"].values
                m = float(np.mean(shifts))
                ci = 1.96 * float(np.std(shifts, ddof=1)) / math.sqrt(len(shifts))
                means.append(m)
                cis.append(ci)
            else:
                means.append(np.nan)
                cis.append(0)

        ax.errorbar(ratios, means, yerr=cis, fmt="o-",
                    color=TOPO_COLORS[topo], linewidth=1.8, markersize=7,
                    capsize=4, label=TOPO_LABELS[topo])

    ax.set_xticks(ratios)
    ax.set_xticklabels([RATIO_LABELS[r] for r in ratios])
    ax.set_xlabel("Minority Ratio", fontsize=10)
    ax.set_ylabel("Mean Shadow EV Shift", fontsize=10)
    ax.set_title("Dose-Response: Minority Ratio Effect on Belief Shift", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.spines["left"].set_color("#AAAAAA")

    fig.tight_layout()
    _save(fig, "fig4_dose_response", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Cross-dataset / cross-model generalization
# ══════════════════════════════════════════════════════════════════════════════

def fig5_generalization(df: pd.DataFrame, out_dir: str):
    print("Generating fig5_generalization ...")

    datasets = sorted(df["dataset"].unique())
    models = sorted(df["model_key"].unique())

    # Skip if only one dataset and one model
    if len(datasets) <= 1 and len(models) <= 1:
        print("  Skipping: only one dataset and model present.")
        return

    n_panels = 0
    if len(datasets) > 1:
        n_panels += 1
    if len(models) > 1:
        n_panels += 1
    if n_panels == 0:
        n_panels = 1

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    fig.patch.set_facecolor("white")
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    # Panel: by dataset
    if len(datasets) > 1:
        ax = axes[panel_idx]
        panel_idx += 1
        positions, labels, colors = [], [], []
        for i, ds in enumerate(datasets):
            sub = df[(df["dataset"] == ds) & (df["topology"] == "fc") &
                     (df["minority_ratio"] == 0.2)]
            matched = sub[["baseline_ev", "shadow_ev"]].dropna()
            if len(matched) > 0:
                shifts = matched["shadow_ev"].values - matched["baseline_ev"].values
                m = float(np.mean(shifts))
                se = float(np.std(shifts, ddof=1)) / math.sqrt(len(shifts))
                ci = 1.96 * se
                ax.barh(i, m, xerr=ci, height=0.6, color=TOPO_COLORS["fc"],
                        capsize=3, alpha=0.8)
                ax.text(m + ci + 0.02, i, f"n={len(shifts)}", va="center", fontsize=8)
            labels.append(ds)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Mean Shadow EV Shift (FC, ratio=0.2)")
        ax.set_title("Cross-Dataset Comparison")
        ax.axvline(0, color="#999999", linestyle="--", linewidth=0.8)
        ax.spines["bottom"].set_color("#AAAAAA")
        ax.spines["left"].set_color("#AAAAAA")

    # Panel: by model
    if len(models) > 1:
        ax = axes[panel_idx]
        for i, model in enumerate(models):
            sub = df[(df["model_key"] == model) & (df["topology"] == "fc") &
                     (df["minority_ratio"] == 0.2)]
            matched = sub[["baseline_ev", "shadow_ev"]].dropna()
            if len(matched) > 0:
                shifts = matched["shadow_ev"].values - matched["baseline_ev"].values
                m = float(np.mean(shifts))
                se = float(np.std(shifts, ddof=1)) / math.sqrt(len(shifts))
                ci = 1.96 * se
                ax.barh(i, m, xerr=ci, height=0.6, color=TOPO_COLORS["star"],
                        capsize=3, alpha=0.8)
                ax.text(m + ci + 0.02, i, f"n={len(shifts)}", va="center", fontsize=8)

        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=9)
        ax.set_xlabel("Mean Shadow EV Shift (FC, ratio=0.2)")
        ax.set_title("Cross-Model Comparison")
        ax.axvline(0, color="#999999", linestyle="--", linewidth=0.8)
        ax.spines["bottom"].set_color("#AAAAAA")
        ax.spines["left"].set_color("#AAAAAA")

    fig.tight_layout()
    _save(fig, "fig5_generalization", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Star topology: hub vs leaf position
# ══════════════════════════════════════════════════════════════════════════════

def fig6_star_position(df: pd.DataFrame, out_dir: str):
    print("Generating fig6_star_position ...")

    star = df[df["topology"] == "star"]
    if len(star) == 0:
        print("  Skipping: no star topology data.")
        return

    ratios = [0.1, 0.2, 0.3]
    pos_labels = {0: "Min. as Leaf", 1: "Min. as Hub"}
    pos_colors = {0: "#6BAED6", 1: "#08306B"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    # Panel 1: CR by position config
    ax = axes[0]
    bar_w = 0.32
    x = np.arange(len(ratios))
    for pi, pos in enumerate([0, 1]):
        offset = (pi - 0.5) * bar_w
        vals = []
        for ratio in ratios:
            sub = star[(star["minority_ratio"] == ratio) &
                       (star["position_config"] == pos)]
            matched = sub[["baseline_ev", "shadow_ev"]].dropna()
            if len(matched) > 0:
                shifts = matched["shadow_ev"].values - matched["baseline_ev"].values
                vals.append(float(np.mean(shifts)))
            else:
                vals.append(0.0)
        ax.bar(x + offset, vals, width=bar_w * 0.9,
               color=pos_colors[pos], label=pos_labels[pos],
               edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([RATIO_LABELS[r] for r in ratios])
    ax.set_xlabel("Minority Ratio")
    ax.set_ylabel("Mean Shadow EV Shift")
    ax.set_title("Star: Gatekeeper Effect")
    ax.legend(fontsize=9, frameon=False)
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.spines["left"].set_color("#AAAAAA")

    # Panel 2: II by position config
    ax2 = axes[1]
    for pi, pos in enumerate([0, 1]):
        offset = (pi - 0.5) * bar_w
        vals = []
        for ratio in ratios:
            sub = star[(star["minority_ratio"] == ratio) &
                       (star["position_config"] == pos)]
            ii_vals = sub["internalization_index"].dropna().values
            vals.append(float(np.median(ii_vals)) if len(ii_vals) > 0 else 0.0)
        ax2.bar(x + offset, vals, width=bar_w * 0.9,
                color=pos_colors[pos], label=pos_labels[pos],
                edgecolor="white", linewidth=0.4)

    ax2.axhline(1.0, color="#999999", linestyle="--", linewidth=0.8, label="Full internalization")
    ax2.set_xticks(x)
    ax2.set_xticklabels([RATIO_LABELS[r] for r in ratios])
    ax2.set_xlabel("Minority Ratio")
    ax2.set_ylabel("Median Internalization Index")
    ax2.set_title("Star: Internalization by Position")
    ax2.legend(fontsize=8, frameon=False)
    ax2.spines["bottom"].set_color("#AAAAAA")
    ax2.spines["left"].set_color("#AAAAAA")

    fig.tight_layout()
    _save(fig, "fig6_star_position", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Prompt-induced vs model-induced equivalence
# ══════════════════════════════════════════════════════════════════════════════

def fig7_condition_equivalence(df: pd.DataFrame, out_dir: str):
    print("Generating fig7_condition_equivalence ...")

    pi_data = df[df["model_condition"] == "prompt_induced"]
    mi_data = df[df["model_condition"] == "model_induced"]

    if len(pi_data) == 0 or len(mi_data) == 0:
        print("  Skipping: need both prompt_induced and model_induced data.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("white")

    # Aggregate per (scenario, topology, ratio): mean shadow shift
    group_cols = ["scenario_id", "topology", "minority_ratio"]
    pi_agg = pi_data.groupby(group_cols).apply(
        lambda g: (g["shadow_ev"].dropna() - g["baseline_ev"].dropna()).mean()
    ).reset_index(name="pi_shift")
    mi_agg = mi_data.groupby(group_cols).apply(
        lambda g: (g["shadow_ev"].dropna() - g["baseline_ev"].dropna()).mean()
    ).reset_index(name="mi_shift")

    merged = pd.merge(pi_agg, mi_agg, on=group_cols, how="inner")
    if len(merged) == 0:
        print("  Skipping: no overlapping conditions.")
        plt.close(fig)
        return

    ax.scatter(merged["pi_shift"], merged["mi_shift"], s=15, alpha=0.5,
               color=TOPO_COLORS["fc"], edgecolors="none")

    lims = [min(merged["pi_shift"].min(), merged["mi_shift"].min()) - 0.2,
            max(merged["pi_shift"].max(), merged["mi_shift"].max()) + 0.2]
    ax.plot(lims, lims, "--", color="#999999", linewidth=1)
    ax.set_xlabel("Prompt-Induced: Mean Shadow Shift")
    ax.set_ylabel("Model-Induced: Mean Shadow Shift")
    ax.set_title("Equivalence: Prompt- vs Model-Induced Misalignment")
    ax.set_aspect("equal", adjustable="datalim")

    # Pearson r annotation
    from scipy import stats
    r_val, p_val = stats.pearsonr(merged["pi_shift"], merged["mi_shift"])
    ax.text(0.05, 0.95, f"r = {r_val:.3f}\np = {p_val:.1e}\nn = {len(merged)}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.9))

    ax.spines["bottom"].set_color("#AAAAAA")
    ax.spines["left"].set_color("#AAAAAA")
    fig.tight_layout()
    _save(fig, "fig7_condition_equivalence", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — Prompt rigidity 2x2 interaction
# ══════════════════════════════════════════════════════════════════════════════

def fig8_prompt_rigidity(df: pd.DataFrame, out_dir: str):
    print("Generating fig8_prompt_rigidity ...")

    strategies = df["prompt_strategy"].unique()
    if len(strategies) <= 1:
        print("  Skipping: only one prompt strategy present.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    strategy_colors = {
        "rigid:rigid": "#08306B",
        "rigid:lenient": "#2171B5",
        "lenient:rigid": "#C44E52",
        "lenient:lenient": "#DD8452",
    }

    # Panel 1: Mean shadow shift by strategy
    ax = axes[0]
    strat_results = []
    for strat in sorted(strategies):
        sub = df[(df["prompt_strategy"] == strat) & (df["topology"] == "fc") &
                 (df["minority_ratio"] == 0.2)]
        matched = sub[["baseline_ev", "shadow_ev"]].dropna()
        if len(matched) > 0:
            shifts = matched["shadow_ev"].values - matched["baseline_ev"].values
            strat_results.append({
                "strategy": strat,
                "mean": float(np.mean(shifts)),
                "ci": 1.96 * float(np.std(shifts, ddof=1)) / math.sqrt(len(shifts)),
                "n": len(shifts),
            })

    if strat_results:
        y_pos = range(len(strat_results))
        for i, sr in enumerate(strat_results):
            color = strategy_colors.get(sr["strategy"], "#999999")
            ax.barh(i, sr["mean"], xerr=sr["ci"], height=0.6,
                    color=color, capsize=3, alpha=0.85)
            ax.text(sr["mean"] + sr["ci"] + 0.02, i,
                    f"n={sr['n']}", va="center", fontsize=8)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels([sr["strategy"] for sr in strat_results], fontsize=9)

    ax.set_xlabel("Mean Shadow EV Shift")
    ax.set_title("Prompt Rigidity: Shadow Shift (FC, ratio=0.2)")
    ax.axvline(0, color="#999999", linestyle="--", linewidth=0.8)
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.spines["left"].set_color("#AAAAAA")

    # Panel 2: II by strategy
    ax2 = axes[1]
    strat_ii = []
    for strat in sorted(strategies):
        sub = df[(df["prompt_strategy"] == strat) & (df["topology"] == "fc") &
                 (df["minority_ratio"] == 0.2)]
        ii_vals = sub["internalization_index"].dropna().values
        if len(ii_vals) > 0:
            strat_ii.append({
                "strategy": strat,
                "median": float(np.median(ii_vals)),
                "n": len(ii_vals),
            })

    if strat_ii:
        y_pos = range(len(strat_ii))
        for i, si in enumerate(strat_ii):
            color = strategy_colors.get(si["strategy"], "#999999")
            ax2.barh(i, si["median"], height=0.6, color=color, alpha=0.85)
            ax2.text(si["median"] + 0.02, i,
                     f"n={si['n']}", va="center", fontsize=8)
        ax2.set_yticks(list(y_pos))
        ax2.set_yticklabels([si["strategy"] for si in strat_ii], fontsize=9)

    ax2.axvline(1.0, color="#999999", linestyle="--", linewidth=0.8,
                label="Full internalization")
    ax2.set_xlabel("Median Internalization Index")
    ax2.set_title("Prompt Rigidity: Internalization (FC, ratio=0.2)")
    ax2.legend(fontsize=8, frameon=False)
    ax2.spines["bottom"].set_color("#AAAAAA")
    ax2.spines["left"].set_color("#AAAAAA")

    fig.tight_layout()
    _save(fig, "fig8_prompt_rigidity", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 9 — Conversion rate grouped bar chart (legacy, updated)
# ══════════════════════════════════════════════════════════════════════════════

def fig9_conversion_rate(df: pd.DataFrame, out_dir: str):
    from .metrics import ev_conversion_rate
    print("Generating fig9_conversion_rate ...")

    ratios = [0.1, 0.2, 0.3]
    bar_w = 0.22
    group_gap = 0.1
    n_topo = len(TOPO_ORDER)
    n_ratio = len(ratios)
    group_w = n_ratio * bar_w + group_gap
    x_centers = np.arange(n_topo) * group_w

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")

    for ri, ratio in enumerate(ratios):
        offsets = (np.arange(n_topo) * group_w
                   + ri * bar_w
                   - ((n_ratio - 1) * bar_w) / 2)
        vals = []
        for topo in TOPO_ORDER:
            sub = df[(df["topology"] == topo) & (df["minority_ratio"] == ratio)]
            matched = sub[["baseline_ev", "shadow_ev"]].dropna()
            if len(matched) > 0:
                cr = ev_conversion_rate(
                    matched["baseline_ev"].values,
                    matched["shadow_ev"].values,
                    threshold=0.5)
                vals.append(cr)
            else:
                vals.append(0.0)

        ax.bar(offsets, vals, width=bar_w * 0.9,
               color=RATIO_COLORS[ratio], label=RATIO_LABELS[ratio],
               edgecolor="white", linewidth=0.4)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([TOPO_LABELS[t] for t in TOPO_ORDER])
    ax.set_ylabel("EV Conversion Rate (threshold=0.5)")
    ax.set_xlabel("Topology")
    ax.set_title("EV-Based Conversion Rate by Topology and Minority Ratio")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend(title="Minority ratio", fontsize=9, title_fontsize=9, frameon=False)
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.spines["left"].set_color("#AAAAAA")

    fig.tight_layout()
    _save(fig, "fig9_conversion_rate", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Generate experiment plots")
    p.add_argument("--experiment", required=True,
                   help="Experiment name (reads from outputs/<name>/).")
    p.add_argument("--results-file", default=None,
                   help="Path to JSONL results file.")
    p.add_argument("--figures-dir", default=None,
                   help="Directory for output figures.")
    p.add_argument("--label", default=None,
                   help="Phase label for plot titles.")
    return p.parse_args()


def cli():
    args = parse_args()

    exp_dir = OUTPUTS_DIR / args.experiment
    results_file = args.results_file or str(exp_dir / "results.jsonl")
    out_dir = args.figures_dir or str(exp_dir / "figures")

    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading trials from {results_file} ...")
    trials = load_trials(results_file)
    print(f"  Loaded {len(trials)} trials.")

    print("Building dataframe ...")
    df = trials_to_dataframe(trials)
    print(f"  {len(df)} aligned agent observations.")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")
    print(f"  Models: {sorted(df['model_key'].unique())}")
    print()

    # Import ev_conversion_rate for fig9
    from .metrics import ev_conversion_rate

    fig1_core_nature(df, out_dir)
    fig2_ii_heatmap(df, out_dir)
    fig3_entropy_trajectories(df, out_dir)
    fig4_dose_response(df, out_dir)
    fig5_generalization(df, out_dir)
    fig6_star_position(df, out_dir)
    fig7_condition_equivalence(df, out_dir)
    fig8_prompt_rigidity(df, out_dir)
    fig9_conversion_rate(df, out_dir)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    cli()
