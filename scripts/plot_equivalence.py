"""
plot_equivalence.py

Redesigned Fig 7: Prompt-Induced vs Model-Induced equivalence.

Two panels:
  Left  — Bland-Altman difference plot:
            x = mean of (PI shift, MI shift) per condition
            y = MI shift − PI shift
            Shaded band = TOST equivalence zone (±δ = 0.2)
            Horizontal line at 0 and at ±δ
            Points coloured by topology

  Right — Scatter (PI vs MI), coloured by topology,
            with identity line and ±δ equivalence band shaded,
            plus 95% CI of mean difference annotated

Usage (from repo root):
  python scripts/plot_equivalence.py

Output:
  outputs/primary_em/figures/fig7_equivalence_redesign.png
"""

from __future__ import annotations
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

from misalignment_contagion.config import OUTPUTS_DIR, N_ROUNDS

# ── style ─────────────────────────────────────────────────────────────────────
TOPO_COLORS = {"fc": "#4C72B0", "circle": "#DD8452", "star": "#55A868", "chain": "#C44E52"}
TOPO_LABELS = {"fc": "FC", "circle": "Circle", "star": "Star", "chain": "Chain"}
TOPO_ORDER  = ["fc", "circle", "star", "chain"]
DELTA       = 0.2   # TOST equivalence margin

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
    "figure.dpi": 150,
})


# ── data loading ──────────────────────────────────────────────────────────────

def load_shifts(path: str) -> dict:
    """Return {(scenario_id, topology, minority_ratio): mean_shadow_ev_shift}."""
    shifts = {}
    with open(path) as f:
        for line in f:
            trial = json.loads(line)
            key = (trial["scenario_id"], trial["topology"], trial["minority_ratio"])
            aligned = [a for a in trial["agents"] if a["role"] == "aligned"]
            vals = []
            for a in aligned:
                b  = a.get("baseline_ev") or _ev_from_probs(a.get("baseline_probs"))
                sh = a.get("shadow_ev")   or _ev_from_probs(a.get("shadow_probs"))
                if b is not None and sh is not None:
                    vals.append(sh - b)
            if vals:
                shifts[key] = float(np.mean(vals))
    return shifts


def _ev_from_probs(probs) -> float | None:
    if not probs:
        return None
    try:
        return sum(float(k) * float(v) for k, v in probs.items())
    except Exception:
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = str(OUTPUTS_DIR / "primary_em" / "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading model-induced (primary_em) ...")
    mi_shifts = load_shifts(
        str(OUTPUTS_DIR / "primary_em" / "synthetic" / "qwen-7b-instruct" / "results.jsonl")
    )
    print("Loading prompt-induced (primary) ...")
    pi_shifts = load_shifts(
        str(OUTPUTS_DIR / "primary" / "synthetic" / "qwen-7b-instruct" / "results.jsonl")
    )

    # Paired conditions
    common_keys = sorted(set(mi_shifts) & set(pi_shifts))
    print(f"Matched conditions: {len(common_keys)}")

    mi = np.array([mi_shifts[k] for k in common_keys])
    pi = np.array([pi_shifts[k] for k in common_keys])
    topos = [k[1] for k in common_keys]

    diff   = mi - pi          # MI − PI
    mean_  = (mi + pi) / 2   # Bland-Altman x-axis

    # TOST stats
    t_stat, p_two = stats.ttest_1samp(diff, popmean=0)
    mean_diff = diff.mean()
    se_diff   = diff.std(ddof=1) / np.sqrt(len(diff))
    ci95_lo   = mean_diff - 1.96 * se_diff
    ci95_hi   = mean_diff + 1.96 * se_diff
    r_val, _  = stats.pearsonr(pi, mi)

    # TOST p-values
    t_low  = (mean_diff - (-DELTA)) / se_diff
    t_high = (mean_diff -   DELTA)  / se_diff
    p_low  = stats.t.sf( t_low,  df=len(diff)-1)
    p_high = stats.t.cdf(t_high, df=len(diff)-1)
    p_tost = max(p_low, p_high)

    print(f"Mean diff (MI-PI): {mean_diff:.4f}  95% CI [{ci95_lo:.4f}, {ci95_hi:.4f}]")
    print(f"TOST p = {p_tost:.4f}  (delta={DELTA})")

    # ── figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("white")

    # ── Panel A: Bland-Altman difference plot ─────────────────────────────────
    for topo in TOPO_ORDER:
        idx = [i for i, t in enumerate(topos) if t == topo]
        ax1.scatter(mean_[idx], diff[idx], s=22, alpha=0.55,
                    color=TOPO_COLORS[topo], edgecolors="none",
                    label=TOPO_LABELS[topo], zorder=3)

    # Equivalence band
    xlim = (mean_.min() - 0.1, mean_.max() + 0.1)
    ax1.fill_between(xlim, -DELTA, DELTA, color="#AAAAAA", alpha=0.12, zorder=1,
                     label=f"Equivalence zone (±{DELTA})")
    ax1.axhline(0,      color="#555555", linewidth=1.0, linestyle="-",  zorder=2)
    ax1.axhline( DELTA, color="#888888", linewidth=0.8, linestyle="--", zorder=2)
    ax1.axhline(-DELTA, color="#888888", linewidth=0.8, linestyle="--", zorder=2)

    # Mean difference ± 95% CI as horizontal band
    ax1.fill_between(xlim, ci95_lo, ci95_hi, color="#4C72B0", alpha=0.15, zorder=2)
    ax1.axhline(mean_diff, color="#4C72B0", linewidth=1.5, linestyle="-", zorder=3,
                label=f"Mean diff = {mean_diff:+.3f}")

    # Annotate TOST result
    equiv_str = "Equivalent ✓" if p_tost < 0.05 else "Not equivalent"
    ax1.text(0.97, 0.97,
             f"Mean diff = {mean_diff:+.3f}\n"
             f"95% CI [{ci95_lo:+.3f}, {ci95_hi:+.3f}]\n"
             f"TOST p = {p_tost:.4f}\n"
             f"δ = {DELTA}  →  {equiv_str}",
             transform=ax1.transAxes, fontsize=8.5, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="#CCCCCC", alpha=0.95))

    ax1.set_xlim(xlim)
    ax1.set_xlabel("Mean shadow EV shift  (PI + MI) / 2", fontsize=10)
    ax1.set_ylabel("Difference  (MI − PI)", fontsize=10)
    ax1.set_title("A   Bland-Altman: MI − PI per condition", fontsize=11, fontweight="bold")
    ax1.spines["bottom"].set_color("#AAAAAA")
    ax1.spines["left"].set_color("#AAAAAA")
    ax1.legend(fontsize=8, frameon=False, loc="lower left",
               handles=[
                   *[mpatches.Patch(color=TOPO_COLORS[t], label=TOPO_LABELS[t])
                     for t in TOPO_ORDER],
                   mpatches.Patch(color="#AAAAAA", alpha=0.4,
                                  label=f"Equiv. zone ±{DELTA}"),
               ])

    # ── Panel B: scatter PI vs MI, coloured by topology ───────────────────────
    for topo in TOPO_ORDER:
        idx = [i for i, t in enumerate(topos) if t == topo]
        ax2.scatter(pi[idx], mi[idx], s=22, alpha=0.55,
                    color=TOPO_COLORS[topo], edgecolors="none",
                    label=TOPO_LABELS[topo], zorder=3)

    lims = (min(pi.min(), mi.min()) - 0.15,
            max(pi.max(), mi.max()) + 0.15)

    # Equivalence band around identity line
    x_band = np.array(lims)
    ax2.fill_between(x_band, x_band - DELTA, x_band + DELTA,
                     color="#AAAAAA", alpha=0.15, zorder=1,
                     label=f"±{DELTA} equivalence band")
    ax2.plot(lims, lims, "--", color="#888888", linewidth=1.0,
             zorder=2, label="Identity (MI = PI)")

    # OLS regression line
    slope, intercept, *_ = stats.linregress(pi, mi)
    x_fit = np.array(lims)
    ax2.plot(x_fit, slope * x_fit + intercept, "-",
             color="#4C72B0", linewidth=1.5, zorder=4,
             label=f"OLS  r = {r_val:.3f}")

    ax2.set_xlim(lims); ax2.set_ylim(lims)
    ax2.set_xlabel("Prompt-Induced: mean shadow EV shift", fontsize=10)
    ax2.set_ylabel("Model-Induced: mean shadow EV shift", fontsize=10)
    ax2.set_title("B   Scatter: PI vs MI by topology", fontsize=11, fontweight="bold")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.spines["bottom"].set_color("#AAAAAA")
    ax2.spines["left"].set_color("#AAAAAA")
    ax2.legend(fontsize=8, frameon=False, loc="upper left")

    fig.suptitle(
        "Prompt-Induced vs Model-Induced Misalignment — Equivalence Analysis\n"
        "Synthetic dataset, Qwen-2.5-7B-Instruct, all topologies × ratios",
        fontsize=12, fontweight="bold", y=1.02
    )
    fig.tight_layout()

    out_path = os.path.join(out_dir, "fig7_equivalence_redesign.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
