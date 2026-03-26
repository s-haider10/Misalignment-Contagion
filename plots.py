"""
plots.py — Generate publication-quality figures for the Misalignment Contagion experiment.

Usage:
  python plots.py                          # default: primary phase
  python plots.py --phase primary_em       # EM phase
  python plots.py --phase primary_em --results-file results/primary_em.jsonl --analysis-dir analysis_em

Figures saved as both PDF and PNG (dpi=300).
"""

import argparse
import json
import os
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

# ── paths (defaults, overridden by CLI) ───────────────────────────────────────
REPO = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(REPO, "results", "primary.jsonl")
CR_TABLE = os.path.join(REPO, "analysis", "cr_table.csv")
DELTA_CC_TABLE = os.path.join(REPO, "analysis", "delta_cc_table.csv")
DTW_TABLE = os.path.join(REPO, "analysis", "dtw_table.csv")
LOGPROB_EV_TABLE = os.path.join(REPO, "analysis", "logprob_ev_table.csv")
OUT_DIR = os.path.join(REPO, "analysis")
PHASE_LABEL = "Prompt-Induced"

# ── style constants ────────────────────────────────────────────────────────────
TOPO_COLORS = {
    "fc":     "#4C72B0",
    "circle": "#DD8452",
    "star":   "#55A868",
    "chain":  "#C44E52",
}
TOPO_LABELS = {"fc": "FC", "circle": "Circle", "star": "Star", "chain": "Chain"}
RATIO_COLORS = {0.1: "#6BAED6", 0.2: "#2171B5", 0.3: "#08306B"}
RATIO_LABELS = {0.1: "10 %", 0.2: "20 %", 0.3: "30 %"}

STAGE_LABELS = ["Baseline", "R0", "R1", "R2", "R3", "R4", "Shadow"]
TOPO_ORDER = ["fc", "circle", "star", "chain"]

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.7,
    "figure.dpi": 150,
})


# ── helpers ───────────────────────────────────────────────────────────────────

def ev(probs: dict) -> float:
    """Expected value on 1-7 scale from a string-keyed probability dict."""
    return sum(int(k) * v for k, v in probs.items())


def save(fig, name: str):
    """Save figure as PDF and PNG to OUT_DIR."""
    base = os.path.join(OUT_DIR, name)
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    print(f"  Saved {base}.pdf  /  .png")
    plt.close(fig)


def load_trials():
    """Load all trials from primary.jsonl into a list of dicts."""
    trials = []
    with open(RESULTS_FILE) as fh:
        for line in fh:
            trials.append(json.loads(line))
    return trials


def aligned_records(trials, topology=None, minority_ratio=None, position_config=None):
    """
    Yield one record per aligned agent, filtered by conditions.
    Each record has keys: topology, minority_ratio, position_config,
    baseline_ev, round_evs (list of 5), shadow_ev.
    """
    for t in trials:
        if topology is not None and t["topology"] != topology:
            continue
        if minority_ratio is not None and t["minority_ratio"] != minority_ratio:
            continue
        if position_config is not None and t["position_config"] != position_config:
            continue
        for a in t["agents"]:
            if a["role"] != "aligned":
                continue
            yield {
                "topology": t["topology"],
                "minority_ratio": t["minority_ratio"],
                "position_config": t["position_config"],
                "baseline_ev": ev(a["baseline_probs"]),
                "round_evs": [ev(rp) for rp in a["round_probs"]],
                "shadow_ev": ev(a["shadow_probs"]),
            }


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Belief trajectory heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig1_trajectory_heatmap(trials):
    print("Generating fig1_trajectory_heatmap …")

    # Build matrix: rows = topologies, cols = 7 stages
    matrix = np.zeros((len(TOPO_ORDER), len(STAGE_LABELS)))

    for ri, topo in enumerate(TOPO_ORDER):
        recs = list(aligned_records(trials, topology=topo,
                                    minority_ratio=0.2, position_config=0))
        if not recs:
            continue
        # baseline
        matrix[ri, 0] = np.mean([r["baseline_ev"] for r in recs])
        # 5 rounds
        for rnd in range(5):
            matrix[ri, rnd + 1] = np.mean([r["round_evs"][rnd] for r in recs])
        # shadow
        matrix[ri, 6] = np.mean([r["shadow_ev"] for r in recs])

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("white")

    # Diverging colormap centred at 4 (neutral)
    vmin, vmax = 1, 7
    center = 4
    # map so that 4 → 0 in a symmetric diverging map
    cmap = plt.cm.RdBu_r  # blue=low (A-side), red=high (B-side)

    im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax,
                   interpolation="nearest")

    # Annotate cells
    for ri in range(len(TOPO_ORDER)):
        for ci in range(len(STAGE_LABELS)):
            val = matrix[ri, ci]
            # Choose text colour for contrast
            mid = (vmin + vmax) / 2
            txt_color = "white" if abs(val - mid) > 1.2 else "black"
            ax.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                    fontsize=8.5, color=txt_color, fontweight="bold")

    ax.set_xticks(range(len(STAGE_LABELS)))
    ax.set_xticklabels(STAGE_LABELS, fontsize=9)
    ax.set_yticks(range(len(TOPO_ORDER)))
    ax.set_yticklabels([TOPO_LABELS[t] for t in TOPO_ORDER], fontsize=9)
    ax.set_xlabel("Stage", fontsize=11)
    ax.set_ylabel("Topology", fontsize=11)
    ax.set_title("Mean Belief Trajectory (logprob EV) — minority ratio 0.2, pos 0",
                 fontsize=11, pad=10)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Mean EV (1=strongly A  →  7=strongly B)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Turn off grid for heatmap
    ax.grid(False)
    ax.spines[:].set_visible(False)

    fig.tight_layout()
    save(fig, "fig1_trajectory_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Conversion rate grouped bar chart
# ══════════════════════════════════════════════════════════════════════════════

def fig2_conversion_rate():
    print("Generating fig2_conversion_rate …")

    df = pd.read_csv(CR_TABLE)
    # Average across position_configs per (topology, minority_ratio)
    agg = (df.groupby(["topology", "minority_ratio"])["cr_delta1"]
             .mean().reset_index())

    ratios = [0.1, 0.2, 0.3]
    n_topo = len(TOPO_ORDER)
    n_ratio = len(ratios)
    bar_w = 0.22
    group_gap = 0.1
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
            row = agg[(agg["topology"] == topo) & (agg["minority_ratio"] == ratio)]
            vals.append(float(row["cr_delta1"].values[0]) if len(row) else 0.0)
        ax.bar(offsets, vals, width=bar_w * 0.9,
               color=RATIO_COLORS[ratio], label=RATIO_LABELS[ratio],
               edgecolor="white", linewidth=0.4)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([TOPO_LABELS[t] for t in TOPO_ORDER], fontsize=9)
    ax.set_ylabel("Conversion Rate (δ ≥ 1)", fontsize=11)
    ax.set_xlabel("Topology", fontsize=11)
    ax.set_title("Conversion Rate by Topology and Minority Ratio", fontsize=11, pad=8)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))

    legend = ax.legend(title="Minority ratio", fontsize=9, title_fontsize=9,
                       frameon=False, loc="upper right")
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.spines["left"].set_color("#AAAAAA")

    fig.tight_layout()
    save(fig, "fig2_conversion_rate")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Logprob distribution shift (violin)
# ══════════════════════════════════════════════════════════════════════════════

def fig3_belief_distribution(trials):
    print("Generating fig3_belief_distribution …")

    recs = list(aligned_records(trials, topology="fc",
                                minority_ratio=0.2, position_config=0))

    baseline = [r["baseline_ev"] for r in recs]
    final    = [r["round_evs"][-1] for r in recs]   # R4
    shadow   = [r["shadow_ev"] for r in recs]

    data_dict = {"Baseline": baseline, "Final Round (R4)": final, "Shadow": shadow}
    labels = list(data_dict.keys())
    data_list = [data_dict[l] for l in labels]

    STAGE_COLORS = ["#6BAED6", "#2171B5", "#C44E52"]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")

    parts = ax.violinplot(data_list, positions=range(len(labels)),
                          showmedians=True, showextrema=True, widths=0.6)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(STAGE_COLORS[i])
        pc.set_alpha(0.55)
        pc.set_edgecolor("#333333")
        pc.set_linewidth(0.8)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color("#333333")
        parts[key].set_linewidth(1.2)

    # Jittered individual points
    rng = np.random.default_rng(42)
    for i, data in enumerate(data_list):
        jitter = rng.uniform(-0.12, 0.12, size=len(data))
        ax.scatter(np.full(len(data), i) + jitter, data,
                   s=4, alpha=0.35, color=STAGE_COLORS[i],
                   edgecolors="none", zorder=3)

    ax.axhline(4, color="#999999", linestyle="--", linewidth=0.9, label="Neutral (4)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Logprob Expected Value (1–7)", fontsize=11)
    ax.set_xlabel("Stage", fontsize=11)
    ax.set_title("Belief Distribution Shift — FC topology, minority ratio 0.2",
                 fontsize=11, pad=8)
    ax.set_ylim(0.5, 7.5)
    ax.legend(fontsize=8, frameon=False)
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.spines["left"].set_color("#AAAAAA")

    fig.tight_layout()
    save(fig, "fig3_belief_distribution")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Compliance-conversion gap dot plot
# ══════════════════════════════════════════════════════════════════════════════

def fig4_delta_cc():
    print("Generating fig4_delta_cc …")

    df = pd.read_csv(DELTA_CC_TABLE)
    ratios = [0.1, 0.2, 0.3]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")

    ax.axvline(0, color="#555555", linestyle="--", linewidth=1.0, zorder=1)

    # y positions: group by topology (4 groups), offset by ratio within group
    y_gap = 1.0   # gap between topology groups
    ratio_step = 0.2

    ytick_pos = []
    ytick_labels = []

    for ti, topo in enumerate(TOPO_ORDER):
        y_base = ti * (len(ratios) * ratio_step + y_gap)
        center_y = y_base + (len(ratios) - 1) * ratio_step / 2
        ytick_pos.append(center_y)
        ytick_labels.append(TOPO_LABELS[topo])

        for ri, ratio in enumerate(ratios):
            sub = df[(df["topology"] == topo) & (df["minority_ratio"] == ratio)]
            if sub.empty:
                continue
            # Average across position_configs for topologies that have multiple
            mean_val = sub["mean_delta_cc"].mean()
            # Pool variance: mean of std (approximate)
            n_total = sub["n"].sum()
            # Pooled SE = sqrt(sum(std^2 * (n-1)) / (N - k)) / sqrt(N)  — approximate
            pooled_var = (sub["std_delta_cc"]**2 * sub["n"]).sum() / n_total
            se = math.sqrt(pooled_var / n_total)
            ci = 1.96 * se

            y = y_base + ri * ratio_step
            ax.errorbar(mean_val, y,
                        xerr=ci, fmt="o",
                        color=RATIO_COLORS[ratio],
                        markersize=6, capsize=3, capthick=1,
                        linewidth=1.2, zorder=3)

    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels, fontsize=9)
    ax.set_xlabel("Mean Δ CC  (compliance − conversion gap)", fontsize=11)
    ax.set_title("Compliance–Conversion Gap by Topology", fontsize=11, pad=8)

    # Grid only on x for this plot
    ax.grid(False)
    ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)

    # Legend for ratio
    handles = [mpatches.Patch(color=RATIO_COLORS[r], label=RATIO_LABELS[r])
               for r in ratios]
    ax.legend(handles=handles, title="Minority ratio",
              fontsize=9, title_fontsize=9, frameon=False, loc="lower right")
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.spines["left"].set_color("#AAAAAA")

    fig.tight_layout()
    save(fig, "fig4_delta_cc")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Round-by-round trajectory line plot
# ══════════════════════════════════════════════════════════════════════════════

def fig5_round_trajectory(trials):
    print("Generating fig5_round_trajectory …")

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")

    x = np.arange(len(STAGE_LABELS))

    for topo in TOPO_ORDER:
        recs = list(aligned_records(trials, topology=topo,
                                    minority_ratio=0.2, position_config=0))
        if not recs:
            continue
        n = len(recs)
        means = np.zeros(len(STAGE_LABELS))
        stds  = np.zeros(len(STAGE_LABELS))

        vals_by_stage = [[] for _ in range(len(STAGE_LABELS))]
        for r in recs:
            vals_by_stage[0].append(r["baseline_ev"])
            for rnd in range(5):
                vals_by_stage[rnd + 1].append(r["round_evs"][rnd])
            vals_by_stage[6].append(r["shadow_ev"])

        for si in range(len(STAGE_LABELS)):
            arr = np.array(vals_by_stage[si])
            means[si] = arr.mean()
            stds[si]  = arr.std(ddof=1)

        se = stds / math.sqrt(n)
        ci = 1.96 * se
        color = TOPO_COLORS[topo]

        ax.plot(x, means, marker="o", markersize=5,
                color=color, linewidth=1.8, label=TOPO_LABELS[topo], zorder=3)
        ax.fill_between(x, means - ci, means + ci,
                        color=color, alpha=0.15, zorder=2)

    ax.axhline(4, color="#999999", linestyle="--", linewidth=0.9, label="Neutral (4)")

    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS, fontsize=9)
    ax.set_ylabel("Mean Logprob EV (1–7)", fontsize=11)
    ax.set_xlabel("Stage", fontsize=11)
    ax.set_title("Belief Trajectory by Topology — minority ratio 0.2, pos 0",
                 fontsize=11, pad=8)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.spines["left"].set_color("#AAAAAA")

    fig.tight_layout()
    save(fig, "fig5_round_trajectory")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Star hub vs leaf bar chart
# ══════════════════════════════════════════════════════════════════════════════

def fig6_star_position():
    print("Generating fig6_star_position …")

    df = pd.read_csv(CR_TABLE)
    star_df = df[df["topology"] == "star"].copy()

    ratios = [0.1, 0.2, 0.3]
    pos_configs = [0, 1]
    pos_labels  = {0: "Min. as Leaf (pos 0)", 1: "Min. as Hub (pos 1)"}
    pos_colors  = {0: "#6BAED6", 1: "#08306B"}

    bar_w = 0.32
    x = np.arange(len(ratios))

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")

    for pi, pos in enumerate(pos_configs):
        offset = (pi - 0.5) * bar_w
        vals = []
        for ratio in ratios:
            row = star_df[(star_df["minority_ratio"] == ratio) &
                          (star_df["position_config"] == pos)]
            vals.append(float(row["cr_delta1"].values[0]) if len(row) else 0.0)
        ax.bar(x + offset, vals, width=bar_w * 0.9,
               color=pos_colors[pos], label=pos_labels[pos],
               edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([RATIO_LABELS[r] for r in ratios], fontsize=9)
    ax.set_xlabel("Minority Ratio", fontsize=11)
    ax.set_ylabel("Conversion Rate (δ ≥ 1)", fontsize=11)
    ax.set_title("Star Topology: Gatekeeper Effect — Hub vs Leaf Position",
                 fontsize=11, pad=8)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.spines["left"].set_color("#AAAAAA")

    fig.tight_layout()
    save(fig, "fig6_star_position")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Logprob EV shift bar chart (final and shadow shifts by topology)
# ══════════════════════════════════════════════════════════════════════════════

def fig7_ev_shift_bars():
    print("Generating fig7_ev_shift_bars …")

    df = pd.read_csv(LOGPROB_EV_TABLE)
    # Filter ratio=0.2, pos=0 for clean comparison
    sub = df[(df["minority_ratio"] == 0.2) & (df["position_config"] == 0)].copy()

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")

    bar_w = 0.32
    x = np.arange(len(TOPO_ORDER))
    shift_colors = {"Final": "#2171B5", "Shadow": "#C44E52"}

    for i, (label, col) in enumerate(
        [("Final", "shift_final"), ("Shadow", "shift_shadow")]
    ):
        offset = (i - 0.5) * bar_w
        vals = []
        for topo in TOPO_ORDER:
            row = sub[sub["topology"] == topo]
            vals.append(float(row[col].values[0]) if len(row) else 0.0)
        ax.bar(x + offset, vals, width=bar_w * 0.9,
               color=shift_colors[label], label=f"{label} shift",
               edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([TOPO_LABELS[t] for t in TOPO_ORDER], fontsize=9)
    ax.set_ylabel("Logprob EV Shift (from baseline)", fontsize=11)
    ax.set_xlabel("Topology", fontsize=11)
    ax.set_title(f"Logprob EV Shift — {PHASE_LABEL}, minority ratio 0.2",
                 fontsize=11, pad=8)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.spines["left"].set_color("#AAAAAA")

    fig.tight_layout()
    save(fig, "fig7_ev_shift_bars")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — Logprob EV shift heatmap (topology × ratio)
# ══════════════════════════════════════════════════════════════════════════════

def fig8_ev_shift_heatmap():
    print("Generating fig8_ev_shift_heatmap …")

    df = pd.read_csv(LOGPROB_EV_TABLE)
    # Use pos=0 for all topologies
    sub = df[df["position_config"] == 0].copy()

    ratios = [0.1, 0.2, 0.3]
    ratio_labels = ["10%", "20%", "30%"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("white")

    for ax_idx, (col, title_suffix) in enumerate(
        [("shift_final", "Final Round"), ("shift_shadow", "Shadow (Private)")]
    ):
        matrix = np.zeros((len(TOPO_ORDER), len(ratios)))
        for ri, ratio in enumerate(ratios):
            for ti, topo in enumerate(TOPO_ORDER):
                row = sub[(sub["topology"] == topo) & (sub["minority_ratio"] == ratio)]
                matrix[ti, ri] = float(row[col].values[0]) if len(row) else 0.0

        ax = axes[ax_idx]
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=matrix.max() * 1.1,
                       interpolation="nearest")

        for ti in range(len(TOPO_ORDER)):
            for ri in range(len(ratios)):
                val = matrix[ti, ri]
                txt_color = "white" if val > matrix.max() * 0.65 else "black"
                ax.text(ri, ti, f"{val:+.2f}", ha="center", va="center",
                        fontsize=9, color=txt_color, fontweight="bold")

        ax.set_xticks(range(len(ratios)))
        ax.set_xticklabels(ratio_labels, fontsize=9)
        ax.set_yticks(range(len(TOPO_ORDER)))
        ax.set_yticklabels([TOPO_LABELS[t] for t in TOPO_ORDER], fontsize=9)
        ax.set_xlabel("Minority Ratio", fontsize=10)
        ax.set_ylabel("Topology", fontsize=10)
        ax.set_title(f"EV Shift — {title_suffix}", fontsize=11, pad=8)
        ax.grid(False)
        ax.spines[:].set_visible(False)

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        cbar.set_label("EV shift", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(f"Logprob EV Shift by Topology and Ratio — {PHASE_LABEL}",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig8_ev_shift_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 9 — Baseline vs Shadow EV scatter (internalization diagnostic)
# ══════════════════════════════════════════════════════════════════════════════

def fig9_baseline_shadow_scatter(trials):
    print("Generating fig9_baseline_shadow_scatter …")

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("white")

    for topo in TOPO_ORDER:
        recs = list(aligned_records(trials, topology=topo,
                                    minority_ratio=0.2, position_config=0))
        if not recs:
            continue
        bx = [r["baseline_ev"] for r in recs]
        sy = [r["shadow_ev"] for r in recs]
        ax.scatter(bx, sy, s=8, alpha=0.35, color=TOPO_COLORS[topo],
                   label=TOPO_LABELS[topo], edgecolors="none")

    ax.plot([1, 7], [1, 7], "--", color="#999999", linewidth=1, label="No shift")
    ax.set_xlabel("Baseline EV", fontsize=11)
    ax.set_ylabel("Shadow EV (private)", fontsize=11)
    ax.set_title(f"Internalization Diagnostic — {PHASE_LABEL}, ratio 0.2",
                 fontsize=11, pad=8)
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(0.5, 7.5)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.spines["left"].set_color("#AAAAAA")

    fig.tight_layout()
    save(fig, "fig9_baseline_shadow_scatter")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Generate experiment plots")
    p.add_argument("--phase", default="primary",
                   choices=["primary", "primary_em"],
                   help="Experiment phase (default: primary).")
    p.add_argument("--results-file", default=None,
                   help="Path to JSONL results file.")
    p.add_argument("--analysis-dir", default=None,
                   help="Directory with analysis CSVs and for output.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Configure paths based on phase
    phase = args.phase
    analysis_dir = args.analysis_dir or os.path.join(REPO, "analysis_em" if phase == "primary_em" else "analysis")
    results_file = args.results_file or os.path.join(REPO, "results", f"{phase}.jsonl")

    RESULTS_FILE = results_file
    CR_TABLE = os.path.join(analysis_dir, "cr_table.csv")
    DELTA_CC_TABLE = os.path.join(analysis_dir, "delta_cc_table.csv")
    DTW_TABLE = os.path.join(analysis_dir, "dtw_table.csv")
    LOGPROB_EV_TABLE = os.path.join(analysis_dir, "logprob_ev_table.csv")
    OUT_DIR = analysis_dir
    PHASE_LABEL = "Emergently Misaligned" if phase == "primary_em" else "Prompt-Induced"

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading trials from {RESULTS_FILE} …")
    trials = load_trials()
    print(f"  Loaded {len(trials)} trials.\n")

    fig1_trajectory_heatmap(trials)
    fig2_conversion_rate()
    fig3_belief_distribution(trials)
    fig4_delta_cc()
    fig5_round_trajectory(trials)
    fig6_star_position()
    fig7_ev_shift_bars()
    fig8_ev_shift_heatmap()
    fig9_baseline_shadow_scatter(trials)

    print(f"\nAll figures saved to {OUT_DIR}")
