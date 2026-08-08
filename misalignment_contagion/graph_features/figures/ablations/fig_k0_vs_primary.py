"""k0 baseline ablation: deliberation drift in the absence of any misaligned minority.

Headline finding: with k=0 misaligned agents (an all-aligned population), aligned
agents still drift substantially toward the misaligned stance during deliberation.
This bounds the share of the primary effect attributable to the minority itself.
"""
from __future__ import annotations
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..fig_style import (apply, style_axes, add_zero_rule, panel_label,
                          FAMILY_COLORS, FAMILY_LABELS, MUTED, GRID)
from ._common import PLOTS_TABLES, ensure_out

warnings.filterwarnings("ignore")

OUT_STEM = "fig_k0_vs_primary"
SRC = PLOTS_TABLES / "k0_vs_primary" / "summary_table.csv"


def build():
    apply()
    out_dir = ensure_out()
    if not SRC.exists():
        print(f"  (skip) {OUT_STEM}: {SRC} not found")
        return None

    df = pd.read_csv(SRC)
    df.to_csv(out_dir / f"{OUT_STEM}_data.csv", index=False)

    # Canonical short labels for k= conditions
    cond_order = ["k0 (no minority)", "k=1 (ratio=0.1)",
                  "k=2 (ratio=0.2)", "k=3 (ratio=0.3)"]
    cond_short = {"k0 (no minority)": "k=0", "k=1 (ratio=0.1)": "k=1",
                  "k=2 (ratio=0.2)": "k=2", "k=3 (ratio=0.3)": "k=3"}
    topologies = ["chain", "circle", "star", "fc"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    panel_label(axes[0], "A", x=-0.22, y=1.07)
    panel_label(axes[1], "B", x=-0.18, y=1.07)
    panel_label(axes[2], "C", x=-0.18, y=1.07)

    # ── Panel A: Δ public EV (final − baseline) per topology × k
    x = np.arange(len(cond_order))
    w = 0.20
    for j, topo in enumerate(topologies):
        vals, sds = [], []
        for c in cond_order:
            sub = df[(df["topology"] == topo) & (df["condition"] == c)]
            if sub.empty:
                vals.append(np.nan); sds.append(0)
            else:
                vals.append(float(sub["delta_ev"].iloc[0]))
                sds.append(float(sub["delta_ev_sd"].iloc[0]))
        sems = [s / np.sqrt(df[(df["topology"] == topo) &
                                 (df["condition"] == c)]["n_agents"].iloc[0])
                if not df[(df["topology"] == topo) &
                            (df["condition"] == c)].empty else 0
                for s, c in zip(sds, cond_order)]
        offset = (j - (len(topologies) - 1) / 2) * w
        axes[0].bar(x + offset, vals, width=w, yerr=sems,
                     color=FAMILY_COLORS[topo],
                     edgecolor="white", linewidth=0.5,
                     label=FAMILY_LABELS[topo],
                     error_kw=dict(ecolor=MUTED, capsize=2, lw=0.5),
                     zorder=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([cond_short[c] for c in cond_order], fontsize=9)
    axes[0].set_ylabel(r"$\Delta$EV public (final $-$ baseline)")
    axes[0].set_title("Public drift", fontsize=10, pad=6)
    add_zero_rule(axes[0], axis="y")
    style_axes(axes[0], grid_axis="y")
    axes[0].legend(fontsize=7.5, ncol=2, loc="lower right",
                   columnspacing=0.8, handletextpad=0.4)

    # ── Panel B: Δ private (shadow − baseline) per topology × k
    for j, topo in enumerate(topologies):
        vals, sems = [], []
        for c in cond_order:
            sub = df[(df["topology"] == topo) & (df["condition"] == c)]
            if sub.empty:
                vals.append(np.nan); sems.append(0)
            else:
                vals.append(float(sub["delta_shadow"].iloc[0]))
                sds_ = float(sub["delta_shadow_sd"].iloc[0])
                n = float(sub["n_agents"].iloc[0])
                sems.append(sds_ / np.sqrt(n))
        offset = (j - (len(topologies) - 1) / 2) * w
        axes[1].bar(x + offset, vals, width=w, yerr=sems,
                     color=FAMILY_COLORS[topo],
                     edgecolor="white", linewidth=0.5,
                     error_kw=dict(ecolor=MUTED, capsize=2, lw=0.5),
                     zorder=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([cond_short[c] for c in cond_order], fontsize=9)
    axes[1].set_ylabel(r"$\Delta$EV private (shadow $-$ baseline)")
    axes[1].set_title("Private drift", fontsize=10, pad=6)
    add_zero_rule(axes[1], axis="y")
    style_axes(axes[1], grid_axis="y")

    # ── Panel C: conversion rate per topology × k
    for j, topo in enumerate(topologies):
        vals = []
        for c in cond_order:
            sub = df[(df["topology"] == topo) & (df["condition"] == c)]
            vals.append(float(sub["conversion_rate"].iloc[0])
                        if not sub.empty else np.nan)
        offset = (j - (len(topologies) - 1) / 2) * w
        axes[2].bar(x + offset, vals, width=w,
                     color=FAMILY_COLORS[topo],
                     edgecolor="white", linewidth=0.5,
                     zorder=3)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([cond_short[c] for c in cond_order], fontsize=9)
    axes[2].set_ylabel("Conversion rate")
    axes[2].set_title("Stance flips", fontsize=10, pad=6)
    style_axes(axes[2], grid_axis="y")

    fig.suptitle("k=0 baseline: deliberation drift without a misaligned minority",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.text(0.5, -0.06,
             "Synthetic, qwen-7b-instruct. At k=0 the population is fully aligned; "
             "remaining drift is intrinsic to deliberation, not minority influence. "
             "k=1..3 add 1–3 misaligned agents (ratio 10–30%).",
             ha="center", va="top", fontsize=8.5, color=MUTED, style="italic")

    fig.tight_layout()
    png = out_dir / f"{OUT_STEM}.png"
    pdf = out_dir / f"{OUT_STEM}.pdf"
    fig.savefig(png, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png.relative_to(png.parents[2])}")
    return df


if __name__ == "__main__":
    build()
