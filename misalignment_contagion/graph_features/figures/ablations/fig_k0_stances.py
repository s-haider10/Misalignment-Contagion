"""k0 baseline stance distribution: where do aligned agents land at k=0..3?

Headline finding: even at k=0 (no misaligned minority), the bulk of aligned
agents end up on the B-side (misaligned-coded stance) at both final and
shadow stages. The minority pushes this fraction further toward 1.0 but
does not create the polarity.
"""
from __future__ import annotations
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..fig_style import (apply, style_axes, panel_label,
                          MUTED, GRID, RULE)
from ._common import PLOTS_TABLES, ensure_out

warnings.filterwarnings("ignore")

OUT_STEM = "fig_k0_stances"
SRC = PLOTS_TABLES / "k0_stances" / "stance_summary.csv"

# Soft palette for stance regions
SIDE_COLORS = {
    "frac_A_side":   "#5a9fd4",  # aligned-coded (blue)
    "frac_neutral":  "#a8a8a8",  # neutral (gray)
    "frac_B_side":   "#e07a5f",  # misaligned-coded (coral)
}
SIDE_LABELS = {
    "frac_A_side":  "A-side (aligned)",
    "frac_neutral": "Neutral",
    "frac_B_side":  "B-side (misaligned)",
}


def _stacked(ax, df_sub, k_order, title):
    bottoms = np.zeros(len(k_order))
    x = np.arange(len(k_order))
    for col in ["frac_A_side", "frac_neutral", "frac_B_side"]:
        vals = [float(df_sub[df_sub["condition"] == k][col].iloc[0])
                if not df_sub[df_sub["condition"] == k].empty else 0
                for k in k_order]
        ax.bar(x, vals, bottom=bottoms,
                color=SIDE_COLORS[col], label=SIDE_LABELS[col],
                edgecolor="white", linewidth=0.6, zorder=3)
        bottoms += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(k_order, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=10, pad=6)
    style_axes(ax, grid_axis="y")


def build():
    apply()
    out_dir = ensure_out()
    if not SRC.exists():
        print(f"  (skip) {OUT_STEM}: {SRC} not found")
        return None

    df = pd.read_csv(SRC)
    df.to_csv(out_dir / f"{OUT_STEM}_data.csv", index=False)

    pooled = df[df["topology"] == "ALL"].copy()
    k_order = ["k=0", "k=1", "k=2", "k=3"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    panel_label(axes[0], "A", x=-0.18, y=1.07)
    panel_label(axes[1], "B", x=-0.13, y=1.07)

    _stacked(axes[0], pooled[pooled["stage"] == "final"], k_order,
              "Final stance (public, end of deliberation)")
    _stacked(axes[1], pooled[pooled["stage"] == "shadow"], k_order,
              "Shadow stance (private elicitation)")
    axes[0].set_ylabel("Fraction of aligned agents")

    handles = [plt.Rectangle((0, 0), 1, 1, color=SIDE_COLORS[c],
                              label=SIDE_LABELS[c])
                for c in ["frac_A_side", "frac_neutral", "frac_B_side"]]
    fig.legend(handles=handles, loc="lower center", ncol=3,
                bbox_to_anchor=(0.5, -0.05), fontsize=9, frameon=False)
    fig.suptitle("k=0 stance distribution: deliberation drift without a minority",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.text(0.5, -0.12,
             "Pooled across topologies (chain, circle, fc, star), synthetic, "
             "qwen-7b-instruct. At k=0, ~82% of aligned agents land on B-side at "
             "final stage; minority adds ~6 pp.",
             ha="center", va="top", fontsize=8.5, color=MUTED, style="italic")

    fig.tight_layout()
    png = out_dir / f"{OUT_STEM}.png"
    pdf = out_dir / f"{OUT_STEM}.pdf"
    fig.savefig(png, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png.relative_to(png.parents[2])}")
    return pooled


if __name__ == "__main__":
    build()
