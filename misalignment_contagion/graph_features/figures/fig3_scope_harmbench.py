"""Figure 3: Scope condition — explicit harm-elicitation breaks the persistence metric.

Panel A: II distributions per dataset on log-x.
Panel B: JSD(shadow, baseline) vs JSD(final, baseline) scatter — shows why II explodes
         on harmbench (denominators ≈ 0).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .fig_style import (
    apply, panel_label, style_axes,
    DATASET_COLORS, DATASET_LABELS, MUTED, GRID, RULE,
)
from .data import FIG_DIR, ensure_dir, PER_AGENT_DATASETS, load_per_agent

warnings.filterwarnings("ignore")
apply()

OUT_PATH = ensure_dir(FIG_DIR) / "fig3_scope_harmbench.png"
OUT_PDF = ensure_dir(FIG_DIR) / "fig3_scope_harmbench.pdf"

DATASETS = ["synthetic", "moral_stories",
            "harmbench_standard", "harmbench_contextual", "harmbench_copyright"]


# We need raw JSD values for panel B; recompute from shard files
RUNS = Path("outputs/graph_features/runs")

from ...metrics import jsd as jsd_metric


def fix_keys(d):
    if d is None:
        return None
    return {int(k): float(v) for k, v in d.items()}


def collect_jsd_pairs(dataset: str, max_rows: int = 30000, seed: int = 0):
    """For each aligned agent: (jsd_final_baseline, jsd_shadow_baseline)."""
    rng = np.random.default_rng(seed)
    pairs = []
    ds_dir = RUNS / dataset
    if not ds_dir.is_dir():
        return np.empty((0, 2))
    for shard in sorted(ds_dir.glob("results.gpu*.jsonl")):
        with open(shard) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    trial = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if trial.get("model_key") != "qwen-7b-instruct":
                    continue
                if trial.get("model_condition") != "model_induced":
                    continue
                for ag in trial["agents"]:
                    if ag.get("role") != "aligned":
                        continue
                    bp = ag.get("baseline_probs")
                    sp_ = ag.get("shadow_probs")
                    rp = ag.get("round_probs")
                    if bp is None or sp_ is None or not rp:
                        continue
                    fin = rp[-1] if isinstance(rp[-1], dict) else None
                    if fin is None:
                        continue
                    bp_i = fix_keys(bp); sp_i = fix_keys(sp_); fin_i = fix_keys(fin)
                    jsd_f = jsd_metric(fin_i, bp_i)
                    jsd_s = jsd_metric(sp_i, bp_i)
                    if jsd_f is None or jsd_s is None:
                        continue
                    pairs.append((jsd_f, jsd_s))
    arr = np.array(pairs)
    if len(arr) > max_rows:
        idx = rng.choice(len(arr), size=max_rows, replace=False)
        arr = arr[idx]
    return arr


# ── Panel A: II histograms ───────────────────────────────────────────
def panel_a(ax):
    for ds in DATASETS:
        df = load_per_agent(ds)
        ii = df["II"].dropna().values
        # Already clipped in saved files; use a log axis
        ii_pos = ii[ii > 0]
        ax.hist(ii_pos, bins=np.logspace(np.log10(0.01),
                                          np.log10(max(ii_pos.max(), 10)),
                                          80),
                color=DATASET_COLORS[ds], alpha=0.55,
                label=DATASET_LABELS[ds], edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("Internalization Index (II)")
    ax.set_ylabel("Count")
    ax.axvline(1.0, color=RULE, linewidth=0.8, linestyle="--", zorder=2)
    ymax = ax.get_ylim()[1]
    ax.text(1.4, ymax * 0.55,
            "II = 1: full internalization\n(private moves as far as public)",
            fontsize=7.5, color=MUTED, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=GRID))
    ax.text(0.012, ymax * 0.35,
            "II → 0: pure compliance / Asch\n(private unchanged from baseline)",
            fontsize=7.5, color=MUTED, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=GRID))
    ax.legend(loc="upper left", fontsize=8.5)
    style_axes(ax, grid_axis="y")
    ax.set_title("II distributions — heavy tail on harmful content",
                 fontsize=10, pad=8)


# ── Panel B: JSD(shadow, baseline) vs JSD(final, baseline) ──────────
def panel_b(ax):
    for ds in DATASETS:
        arr = collect_jsd_pairs(ds, max_rows=12000, seed=hash(ds) & 0xFFFF)
        if len(arr) == 0:
            continue
        jf = arr[:, 0]; js = arr[:, 1]
        ax.scatter(jf, js, s=4, color=DATASET_COLORS[ds],
                    alpha=0.18, label=DATASET_LABELS[ds],
                    edgecolor="none", zorder=2)
    # Reference: II = 1 corresponds to jsd_shadow == jsd_final
    lim_lo, lim_hi = 1e-5, 1.0
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            color=RULE, linewidth=0.9, linestyle="--", zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel(r"JSD(final, baseline)  [II denominator]")
    ax.set_ylabel(r"JSD(shadow, baseline)  [II numerator]")
    leg = ax.legend(loc="upper left", fontsize=8.5,
                     markerscale=4, handletextpad=0.3)
    for h in leg.legend_handles:
        h.set_alpha(0.9)
    style_axes(ax, grid_axis="both")
    ax.text(0.97, 0.05,
            "HarmBench points cluster\nat near-zero denominator\n→ II ratio explodes",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=MUTED, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=GRID))
    ax.set_title("Why the metric collapses on harmbench",
                 fontsize=10, pad=8)


def build():
    fig = plt.figure(figsize=(11, 4.6))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    panel_a(ax_a)
    panel_b(ax_b)

    panel_label(ax_a, "A")
    panel_label(ax_b, "B")

    fig.savefig(OUT_PATH, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_PATH.relative_to(OUT_PATH.parents[2])}")


if __name__ == "__main__":
    build()
