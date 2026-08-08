"""Bimodality diagnostic: how often does an aligned agent's shadow distribution
put non-trivial mass on BOTH the misaligned-coded stance AND the aligned-coded
stance simultaneously?

Headline finding: bimodal stance sampling is rare on synthetic / moral_stories
/ harmbench_copyright (≤0.5% under any threshold) but materially present on
harmbench_standard and harmbench_contextual (3–12% loose-threshold) — explaining
why shadow-EV estimates are noisier on those scope-condition datasets.
"""
from __future__ import annotations
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..fig_style import (apply, style_axes, panel_label,
                          DATASET_COLORS, DATASET_LABELS, DATASET_ORDER,
                          MUTED, GRID, RULE)
from ._common import PLOTS_TABLES, ensure_out

warnings.filterwarnings("ignore")

OUT_STEM = "fig_bimodality"
SRC = PLOTS_TABLES / "bimodality_diagnostic.csv"

THRESHOLDS = [
    ("frac_bimodal_strict_>0.3", "Strict\n(min(P1,P7)>0.3)"),
    ("frac_bimodal_loose_>0.2",  "Loose\n(min(P1,P7)>0.2)"),
    ("frac_bimodal_soft_>0.1",   "Soft\n(min(P1,P7)>0.1)"),
]


def build():
    apply()
    out_dir = ensure_out()
    if not SRC.exists():
        print(f"  (skip) {OUT_STEM}: {SRC} not found")
        return None

    df = pd.read_csv(SRC)
    df.to_csv(out_dir / f"{OUT_STEM}_data.csv", index=False)

    # Use primary (own-last) variant only
    primary = df[df["variant"] == "own-last"].copy()
    datasets = [d for d in DATASET_ORDER if d in primary["dataset"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    panel_label(axes[0], "A", x=-0.18, y=1.07)
    panel_label(axes[1], "B", x=-0.13, y=1.07)

    # ── Panel A: bimodality at each threshold, per dataset
    x = np.arange(len(THRESHOLDS))
    n_d = len(datasets)
    w = 0.78 / n_d
    for j, ds in enumerate(datasets):
        sub = primary[primary["dataset"] == ds]
        if sub.empty:
            continue
        vals = [100.0 * float(sub[col].iloc[0]) for col, _ in THRESHOLDS]
        offset = (j - (n_d - 1) / 2) * w
        axes[0].bar(x + offset, vals, width=w,
                     color=DATASET_COLORS[ds],
                     edgecolor="white", linewidth=0.6,
                     label=DATASET_LABELS[ds], zorder=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([lbl for _, lbl in THRESHOLDS], fontsize=8.5)
    axes[0].set_ylabel("Fraction bimodal (%)")
    axes[0].set_title("Bimodal stance mass on shadow probe", fontsize=10, pad=6)
    style_axes(axes[0], grid_axis="y")
    axes[0].legend(fontsize=7.5, ncol=2, loc="upper right",
                   columnspacing=0.8, handletextpad=0.4)

    # ── Panel B: mean P1, P4, P7 per dataset (the underlying distribution)
    components = [("mean_P1", "P1 (aligned-coded)"),
                  ("mean_P4", "P4 (neutral)"),
                  ("mean_P7", "P7 (misaligned-coded)")]
    x2 = np.arange(len(components))
    for j, ds in enumerate(datasets):
        sub = primary[primary["dataset"] == ds]
        if sub.empty:
            continue
        vals = [float(sub[col].iloc[0]) for col, _ in components]
        offset = (j - (n_d - 1) / 2) * w
        axes[1].bar(x2 + offset, vals, width=w,
                     color=DATASET_COLORS[ds],
                     edgecolor="white", linewidth=0.6, zorder=3)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([lbl for _, lbl in components], fontsize=8.5)
    axes[1].set_ylabel("Mean probability mass")
    axes[1].set_title("Mean shadow probability at canonical stances",
                       fontsize=10, pad=6)
    style_axes(axes[1], grid_axis="y")

    fig.suptitle("Bimodal stance sampling on aligned-agent shadow elicitation",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.text(0.5, -0.06,
             "Primary (own-last) variant, all 5 datasets. Bimodality is a sampling "
             "artifact where shadow probe places joint mass on stance 1 (aligned-coded) "
             "and stance 7 (misaligned-coded). Rare on synthetic / moral_stories / "
             "harmbench_copyright; material on harmbench_standard and contextual, "
             "consistent with refusal-anchored shadow noise on those datasets.",
             ha="center", va="top", fontsize=8.5, color=MUTED, style="italic",
             wrap=True)

    fig.tight_layout()
    png = out_dir / f"{OUT_STEM}.png"
    pdf = out_dir / f"{OUT_STEM}.pdf"
    fig.savefig(png, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png.relative_to(png.parents[2])}")
    return primary


if __name__ == "__main__":
    build()
