"""Figure 5: Structural manifold — 151 motif-distinct graphs span 7 families.

PCA (or UMAP if available) projection of the 32-D feature space.
Color = family, size = n_agents, marker shape = n_misaligned.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .fig_style import (
    apply, panel_label, style_axes,
    FAMILY_ORDER, FAMILY_COLORS, FAMILY_LABELS, MUTED, GRID,
)
from .data import FIG_DIR, ensure_dir, load_manifest_features

warnings.filterwarnings("ignore")
apply()

OUT_PATH = ensure_dir(FIG_DIR) / "fig5_manifold.png"
OUT_PDF = ensure_dir(FIG_DIR) / "fig5_manifold.pdf"

MARKER_BY_K = {1: "o", 2: "s", 3: "^"}


def try_umap(X):
    """Return UMAP if installed; else None."""
    try:
        import umap  # noqa
        reducer = umap.UMAP(n_components=2, random_state=42,
                            n_neighbors=15, min_dist=0.25)
        return reducer.fit_transform(X), "UMAP"
    except Exception:
        return None, None


def build():
    m_full = load_manifest_features()
    print(f"Manifest: {len(m_full)} graphs, {len(m_full.columns)} columns")
    # Drop very large graphs (n_agents > 20) from MAIN panel — they're outliers
    # that visually dominate the projection. Reported separately in caption.
    outlier_mask = m_full["n_agents"] > 20
    m = m_full[~outlier_mask].reset_index(drop=True)
    n_outliers = int(outlier_mask.sum())
    print(f"  excluding {n_outliers} graphs with n_agents > 20 from main projection")

    drop = {"graph_id", "family", "placement_summary",
            "minority_ratio", "n_agents", "n_misaligned"}
    topo_cols = [c for c in m.columns
                  if c not in drop and np.issubdtype(m[c].dtype, np.number)]
    X = StandardScaler().fit_transform(m[topo_cols].values)

    embedding, method = try_umap(X)
    if embedding is None:
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(X)
        method = "PCA"
        explained = pca.explained_variance_ratio_
        xlabel = f"PC1 ({explained[0]:.0%} var.)"
        ylabel = f"PC2 ({explained[1]:.0%} var.)"
    else:
        xlabel = "UMAP 1"
        ylabel = "UMAP 2"

    fig, ax = plt.subplots(figsize=(14, 6.5))
    plt.subplots_adjust(right=0.72)

    # Plot per family × n_misaligned combination
    for fam in FAMILY_ORDER:
        sub = m[m["family"] == fam].copy()
        if sub.empty:
            continue
        for k in sorted(sub["n_misaligned"].unique()):
            mask = sub["n_misaligned"] == k
            sizes = sub.loc[mask, "n_agents"].values * 5.5  # n_agents -> point area
            ax.scatter(embedding[sub.index[mask], 0],
                       embedding[sub.index[mask], 1],
                       s=sizes, c=FAMILY_COLORS[fam],
                       marker=MARKER_BY_K.get(k, "o"),
                       alpha=0.78, edgecolor="white", linewidth=0.6,
                       zorder=3)

    # Custom legends (family color + n_misaligned shape + size scale)
    fam_handles = [plt.Line2D([], [], marker="o", linestyle="",
                               markersize=8, markeredgecolor="white",
                               markeredgewidth=0.6,
                               markerfacecolor=FAMILY_COLORS[f],
                               label=FAMILY_LABELS[f])
                    for f in FAMILY_ORDER if f in m["family"].unique()]
    leg1 = ax.legend(handles=fam_handles, title="Family",
                      loc="upper left", fontsize=8.5, title_fontsize=9,
                      bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.add_artist(leg1)

    shape_handles = [plt.Line2D([], [], marker=MARKER_BY_K[k], linestyle="",
                                 markersize=8, color=MUTED,
                                 markerfacecolor="white", markeredgecolor=MUTED,
                                 label=f"k = {k}")
                      for k in [1, 2, 3]]
    leg2 = ax.legend(handles=shape_handles, title="# misaligned",
                      loc="upper left", fontsize=8.5, title_fontsize=9,
                      bbox_to_anchor=(1.02, 0.55), borderaxespad=0.0)
    ax.add_artist(leg2)

    # Show min, median, max from the actually-plotted n values
    sizes_shown = sorted(set(int(n) for n in m["n_agents"].unique()))
    if len(sizes_shown) <= 4:
        sizes_legend = sizes_shown
    else:
        # min, ~1/3, ~2/3, max
        sizes_legend = [sizes_shown[0],
                         sizes_shown[len(sizes_shown) // 3],
                         sizes_shown[2 * len(sizes_shown) // 3],
                         sizes_shown[-1]]
        sizes_legend = sorted(set(sizes_legend))
    size_handles = [plt.Line2D([], [], marker="o", linestyle="",
                                markersize=np.sqrt(n * 5.5),
                                color=MUTED, markerfacecolor="lightgray",
                                markeredgecolor=MUTED, label=f"n = {n}")
                     for n in sizes_legend]
    leg3 = ax.legend(handles=size_handles, title="# agents",
              loc="upper left", fontsize=8.5, title_fontsize=9,
              bbox_to_anchor=(1.02, 0.25), borderaxespad=0.0)
    extra_artists = [leg1, leg2, leg3]

    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    title = (f"{method} projection of the 32-dim topology feature space "
             f"({len(m)} graphs)")
    if n_outliers > 0:
        title += f"\n{n_outliers} large graphs (n>20) excluded from projection"
    ax.set_title(title, fontsize=10, pad=10)
    style_axes(ax, grid_axis="both")

    fig.savefig(OUT_PATH, facecolor="white", bbox_inches="tight",
                bbox_extra_artists=extra_artists)
    fig.savefig(OUT_PDF, facecolor="white", bbox_inches="tight",
                bbox_extra_artists=extra_artists)
    plt.close(fig)
    print(f"wrote {OUT_PATH.relative_to(OUT_PATH.parents[2])}")


if __name__ == "__main__":
    build()
