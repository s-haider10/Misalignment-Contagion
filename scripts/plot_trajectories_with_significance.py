"""
plot_trajectories_with_significance.py

Reproduces the EV-logprob, Shannon-entropy, and Δ-stance trajectory figures
(5 datasets × 4 topologies) and overlays significance stars on every stage
whose mean differs from baseline using a paired one-sample t-test
(H0: mean shift from baseline = 0).

Correction: Bonferroni across the 6 non-baseline stages (R1–R5 + Shadow)
per topology × dataset cell.

Stars:
  *   p < 0.05  (after correction)
  **  p < 0.01
  *** p < 0.001

Usage (from repo root):
  python scripts/plot_trajectories_with_significance.py

Output:
  outputs/primary_em/figures/fig_ev_sig.png
  outputs/primary_em/figures/fig_entropy_sig.png
  outputs/primary_em/figures/fig_delta_stance_sig.png
"""

from __future__ import annotations

import sys
import os
import warnings

warnings.filterwarnings("ignore")

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from misalignment_contagion.config import OUTPUTS_DIR, N_ROUNDS
from misalignment_contagion.analyze import load_trials, trials_to_dataframe

# ── constants (match plots.py) ────────────────────────────────────────────────

TOPO_COLORS  = {"fc": "#4C72B0", "circle": "#DD8452", "star": "#55A868", "chain": "#C44E52"}
TOPO_LABELS  = {"fc": "FC", "circle": "Circle", "star": "Star", "chain": "Chain"}
TOPO_ORDER   = ["fc", "circle", "star", "chain"]

DATASET_LABELS = {
    "synthetic":            "Synthetic",
    "moral_stories":        "Moral Stories",
    "harmbench_standard":   "HarmBench-Std",
    "harmbench_contextual": "HarmBench-Ctx",
    "harmbench_copyright":  "HarmBench-Cpy",
}
DATASET_ORDER = [
    "synthetic", "moral_stories",
    "harmbench_standard", "harmbench_contextual", "harmbench_copyright",
]

STAGE_LABELS = ["Baseline"] + [f"R{i}" for i in range(1, N_ROUNDS + 1)] + ["Shadow"]
N_STAGES     = len(STAGE_LABELS)          # 7
N_TEST_STAGES = N_STAGES - 1              # 6 (R1–R5 + Shadow; baseline is reference)

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


# ── significance helpers ──────────────────────────────────────────────────────

def star(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def significance_stars(trajs: np.ndarray, alpha: float = 0.05) -> list[str]:
    """
    For each non-baseline stage (columns 1..N_STAGES-1), run a one-sample
    t-test on (stage_value - baseline_value) against mu=0.
    Apply Bonferroni correction over N_TEST_STAGES comparisons.
    Returns list of star strings, length N_STAGES (index 0 is always "").
    """
    stars = [""]  # baseline: no test
    corrected_alpha = alpha / N_TEST_STAGES
    baseline = trajs[:, 0]
    for s in range(1, N_STAGES):
        shifts = trajs[:, s] - baseline
        if len(shifts) < 2:
            stars.append("")
            continue
        t_stat, p_two = stats.ttest_1samp(shifts, popmean=0)
        p_one = p_two / 2 if t_stat > 0 else 1.0  # one-sided: shift > 0
        corrected_p = min(p_one * N_TEST_STAGES, 1.0)
        stars.append(star(corrected_p))
    return stars


# ── extraction functions ──────────────────────────────────────────────────────

def extract_ev(row) -> list[float] | None:
    b  = row.get("baseline_ev")
    sh = row.get("shadow_ev")
    rs = row.get("round_evs")
    if b is None or sh is None or not isinstance(rs, list) or len(rs) != N_ROUNDS:
        return None
    if any(v is None for v in rs):
        return None
    return [b] + list(rs) + [sh]


def extract_entropy(row) -> list[float] | None:
    traj = row.get("entropy_trajectory")
    if not isinstance(traj, list) or len(traj) != N_ROUNDS + 2:
        return None
    if any(v is None for v in traj):
        return None
    return list(traj)


def extract_delta_stance(row) -> list[float] | None:
    b  = row.get("baseline_stance")
    sh = row.get("shadow_stance")
    rs = row.get("round_stances")
    if b is None or sh is None or not isinstance(rs, list) or len(rs) != N_ROUNDS:
        return None
    if any(v is None for v in [b, sh] + list(rs)):
        return None
    traj = [float(b)] + [float(v) for v in rs] + [float(sh)]
    return [v - traj[0] for v in traj]


# ── core plotting function ────────────────────────────────────────────────────

def make_figure(sub, extract_fn, ylabel, suptitle, filename, out_dir,
                zero_ref=False):
    x    = np.arange(N_STAGES)
    R4_IDX = 4   # Baseline=0, R1=1, R2=2, R3=3, R4=4

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.8), sharey=True)
    fig.patch.set_facecolor("white")

    # Horizontal jitter so 4 topology stars at R4 don't overlap
    # fc, circle, star, chain → offsets around x=4
    TOPO_X_OFFSET = {topo: (i - 1.5) * 0.12 for i, topo in enumerate(TOPO_ORDER)}

    for ci, dataset in enumerate(DATASET_ORDER):
        ax      = axes[ci]
        ds_sub  = sub[sub["dataset"] == dataset]
        any_data = False
        topo_results = []

        for topo in TOPO_ORDER:
            color    = TOPO_COLORS[topo]
            topo_sub = ds_sub[ds_sub["topology"] == topo]

            trajs = []
            for _, row in topo_sub.iterrows():
                t = extract_fn(row)
                if t is not None:
                    trajs.append(t)

            if not trajs:
                topo_results.append(None)
                continue

            any_data = True
            arr    = np.array(trajs)
            means  = arr.mean(axis=0)
            sds    = arr.std(axis=0, ddof=1)
            sstars = significance_stars(arr)
            topo_results.append((topo, color, means, sds, sstars))

            ax.errorbar(x, means, yerr=sds, fmt="o-", color=color,
                        label=TOPO_LABELS[topo], linewidth=1.8, markersize=4,
                        capsize=2.5, capthick=1.0, elinewidth=0.8, zorder=3)
            ax.fill_between(x, means - sds, means + sds,
                            color=color, alpha=0.08, zorder=2)

        # Draw stars at R4 — just above each topology's own marker.
        # Sort by mean at R4 so stars for close lines don't collide;
        # apply a small minimum vertical gap between adjacent stars.
        SMALL_OFFSET = 0.12   # base gap above the marker point
        MIN_VERT_GAP = 0.15   # minimum vertical separation between stars

        valid = [(r[2][R4_IDX], r) for r in topo_results
                 if r is not None and r[3][R4_IDX]]
        valid.sort(key=lambda t: t[0])  # ascending by mean at R4

        placed_y = []  # track y positions already used
        for mean_r4, result in valid:
            topo, color, means, sds, sstars = result
            s = sstars[R4_IDX]
            if not s:
                continue
            xpos = x[R4_IDX] + TOPO_X_OFFSET[topo]
            ypos = mean_r4 + SMALL_OFFSET
            # Push up if too close to an already-placed star
            for py in placed_y:
                if abs(ypos - py) < MIN_VERT_GAP:
                    ypos = py + MIN_VERT_GAP
            placed_y.append(ypos)
            ax.text(xpos, ypos, s, ha="center", va="bottom",
                    fontsize=8, color=color, fontweight="bold", zorder=5)

        # Shadow separator
        ax.axvline(x[-2] + 0.5, color="#CCCCCC", linestyle=":", linewidth=0.9)
        if zero_ref:
            ax.axhline(0, color="#888888", linestyle="-", linewidth=0.8, zorder=1)

        ax.set_xticks(x)
        ax.set_xticklabels(STAGE_LABELS, fontsize=7.5, rotation=45, ha="right")
        ax.set_title(DATASET_LABELS.get(dataset, dataset),
                     fontsize=11, fontweight="bold")
        ax.spines["bottom"].set_color("#AAAAAA")
        ax.spines["left"].set_color("#AAAAAA")
        if ci == 0:
            ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Stage", fontsize=9)

        if not any_data:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="#999999")

    # Topology legend on last panel
    axes[-1].legend(fontsize=8.5, frameon=False, loc="lower right")

    # Significance legend as a small text box on first panel
    sig_text = "Significance at R4\n* p<0.05\n** p<0.01\n*** p<0.001\n(one-sided vs baseline,\nBonferroni-corrected)"
    axes[0].text(0.02, 0.98, sig_text, transform=axes[0].transAxes,
                 fontsize=7, va="top", ha="left", color="#444444",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                           edgecolor="#CCCCCC", alpha=0.9))

    fig.suptitle(
        f"{suptitle} — Qwen-2.5-7B-Instruct, 20% Minority, All Topologies\n"
        "Error bars = ±1 SD (aligned agents only)",
        fontsize=12, fontweight="bold", y=1.02
    )
    fig.tight_layout()

    path = os.path.join(out_dir, filename + ".png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    exp_dir = OUTPUTS_DIR / "primary_em"
    out_dir = str(exp_dir / "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading trials ...")
    trials = load_trials(str(exp_dir / "**" / "results.jsonl"))
    print(f"  {len(trials)} trials loaded.")

    print("Building dataframe ...")
    df = trials_to_dataframe(trials)
    print(f"  {len(df)} aligned agent rows.")

    sub = df[
        (df["model_key"]        == "qwen-7b-instruct") &
        (df["model_condition"]  == "model_induced") &
        (df["minority_ratio"]   == 0.2) &
        (df["position_config"]  == 0) &
        (df["prompt_strategy"]  == "rigid:rigid")
    ].copy()

    print("\nGenerating EV figure ...")
    make_figure(sub, extract_ev,
                ylabel="Mean Logprob EV (1–7)",
                suptitle="Logprob Expected Value",
                filename="fig_ev_sig",
                out_dir=out_dir,
                zero_ref=False)

    print("Generating entropy figure ...")
    make_figure(sub, extract_entropy,
                ylabel="Shannon Entropy (bits)",
                suptitle="Belief Entropy",
                filename="fig_entropy_sig",
                out_dir=out_dir,
                zero_ref=False)

    print("Generating Δ stance figure ...")
    make_figure(sub, extract_delta_stance,
                ylabel="Δ Stance (relative to baseline)",
                suptitle="Stance Shift (Δ)",
                filename="fig_delta_stance_sig",
                out_dir=out_dir,
                zero_ref=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
