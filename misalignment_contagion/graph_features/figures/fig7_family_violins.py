"""Figure 7 (optional): Family-level effect breakdown.

Violin plots of graph-level mean II per topology family, ordered by overall
median. Same violin encoding for all families regardless of n.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .fig_style import (
    apply, panel_label, style_axes,
    FAMILY_COLORS, FAMILY_LABELS, DATASET_COLORS, DATASET_LABELS,
    MUTED, GRID, RULE,
)
from .data import FIG_DIR, ensure_dir, all_datasets_per_agent, graph_level_from_per_agent

warnings.filterwarnings("ignore")
apply()

OUT_PATH = ensure_dir(FIG_DIR) / "fig7_family_violins.png"
OUT_PDF = ensure_dir(FIG_DIR) / "fig7_family_violins.pdf"


def build():
    pa = all_datasets_per_agent()
    rows = []
    # All harmbench variants are scope-condition cases (see Fig 3) — exclude
    # from the family-level breakdown which is about normative-content ordering.
    NORMATIVE = {"synthetic", "moral_stories"}
    for ds, df in pa.items():
        if ds not in NORMATIVE:
            continue
        gl = graph_level_from_per_agent(df, ds)
        for _, r in gl.iterrows():
            rows.append({"dataset": ds, "family": r["family"],
                         "mean_II": r["y_mean_II"]})
    df_all = pd.DataFrame(rows)
    if df_all.empty:
        print("(no data)")
        return

    # Order families by overall median mean_II
    family_order = (df_all.groupby("family")["mean_II"]
                          .median().sort_values(ascending=False).index.tolist())

    fig, ax = plt.subplots(figsize=(10, 5))
    datasets = sorted(df_all["dataset"].unique())
    n_d = len(datasets)
    width = 0.78 / n_d
    x = np.arange(len(family_order))

    for j, ds in enumerate(datasets):
        sub = df_all[df_all["dataset"] == ds]
        color = DATASET_COLORS[ds]
        for i, fam in enumerate(family_order):
            vals = sub[sub["family"] == fam]["mean_II"].values
            if len(vals) == 0:
                continue
            px = i + (j - (n_d - 1) / 2) * width

            parts = ax.violinplot([vals], positions=[px],
                                    widths=width * 0.9,
                                    showmeans=False, showmedians=True,
                                    showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(color); body.set_alpha(0.55)
                body.set_edgecolor(color); body.set_linewidth(0.8)
            if "cmedians" in parts:
                parts["cmedians"].set_color(color)
                parts["cmedians"].set_linewidth(1.6)

    # Per-family medians annotated below the axis
    overall_meds = df_all.groupby("family")["mean_II"].median()
    ax.set_xticks(x)
    ax.set_xticklabels([f"{FAMILY_LABELS[f]}\nmed={overall_meds[f]:.2f}"
                         for f in family_order], fontsize=8.5)
    ax.set_ylabel("Graph-level mean II")
    ax.set_title("Mean II per topology family (synthetic + moral_stories)",
                 fontsize=10, pad=8)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=DATASET_COLORS[ds],
                               alpha=0.7, label=DATASET_LABELS[ds])
                for ds in datasets]
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    style_axes(ax, grid_axis="y")

    fig.savefig(OUT_PATH, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_PATH.relative_to(OUT_PATH.parents[2])}")


if __name__ == "__main__":
    build()
