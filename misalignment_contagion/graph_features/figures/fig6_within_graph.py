"""Figure 6: Within-graph agent heterogeneity (ALL 7 topology families).

Produces two complementary figures:
  fig6a_persistence.png   — aligned nodes colored by predicted Internalization
                            Index (II). Asks: who ends up internalizing peer
                            pressure (persistent change)?
  fig6b_magnitude.png     — aligned nodes colored by predicted shadow EV shift
                            (raw |shift|). Asks: who gets pushed the farthest
                            in the first place (regardless of persistence)?

Both share the SAME colormap so you can compare 6a vs 6b directly:
  low value  → soft blue   (≈ 'safe' agent)
  high value → soft coral  (≈ 'affected' agent)

Colors are globally normalized to the same vmin/vmax across all 7 subplots
within each figure, so absolute position in colorspace is comparable across
topology families.

Misaligned nodes are outlined in a heavier coral edge with white fill so they
read as "sources" rather than "destinations" on the colormap.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .fig_style import (
    apply, panel_label, ROLE_COLORS, MUTED, GRID, FAMILY_LABELS,
    DATASET_COLORS,
)
from .data import FIG_DIR, ensure_dir, load_manifest_graphs, load_per_agent
from .stats_helpers import AGENT_FEATURE_COLS

warnings.filterwarnings("ignore")
apply()

OUT_6A_PNG = ensure_dir(FIG_DIR) / "fig6a_persistence.png"
OUT_6A_PDF = ensure_dir(FIG_DIR) / "fig6a_persistence.pdf"
OUT_6B_PNG = ensure_dir(FIG_DIR) / "fig6b_magnitude.png"
OUT_6B_PDF = ensure_dir(FIG_DIR) / "fig6b_magnitude.pdf"

# ── Color scheme ────────────────────────────────────────────────────
# Soft green → soft coral. Low = green (safe), high = coral (affected).
# Blue removed to avoid confusion with the standalone "synthetic" color
# used elsewhere in the paper.
PALETTE_STOPS = [
    (0.00, "#6cbf6c"),   # soft green   (low)
    (1.00, "#e07a5f"),   # soft coral   (high)
]
SHARED_CMAP = LinearSegmentedColormap.from_list(
    "shared_gr", [(p, c) for p, c in PALETTE_STOPS]
)

# Pick one representative graph per family
FAMILY_PICK = {
    "chain":       lambda g: g["n_agents"] == 12 and g["n_misaligned"] == 2,
    "circle":      lambda g: g["n_agents"] == 10 and g["n_misaligned"] == 2,
    "star":        lambda g: g["n_agents"] in (8, 10) and g["n_misaligned"] == 1,
    "fc":          lambda g: g["n_agents"] == 8 and g["n_misaligned"] in (1, 2),
    "tree":        lambda g: g["n_agents"] in (12, 15) and g["n_misaligned"] in (2, 3),
    "small_world": lambda g: g["n_agents"] == 10 and g["n_misaligned"] == 3,
    "sparse_fc":   lambda g: g["n_agents"] in (10, 12) and g["n_misaligned"] == 2,
}


def fit_model(df_ag, target):
    """OLS predicting `target` from the 9 agent-level features (standardized)."""
    sub = df_ag.dropna(subset=[target] + AGENT_FEATURE_COLS).reset_index(drop=True)
    scaler = StandardScaler().fit(sub[AGENT_FEATURE_COLS].values)
    X = scaler.transform(sub[AGENT_FEATURE_COLS].values)
    y = sub[target].values
    m = LinearRegression().fit(X, y)
    return m, scaler


def per_agent_features(G_dir, n, pos, misaligned_set, centr_cache):
    G_und = G_dir.to_undirected()
    if pos in misaligned_set:
        dist = 0
    else:
        ds = nx.single_source_shortest_path_length(G_und, pos)
        md = [d for p, d in ds.items() if p in misaligned_set]
        dist = min(md) if md else n
    one_hop = set(G_und.neighbors(pos))
    two_hop = set(one_hop)
    for nb in list(one_hop):
        two_hop.update(G_und.neighbors(nb))
    two_hop.discard(pos)
    n_mis_1 = len(one_hop & misaligned_set)
    n_n = len(one_hop)
    return {
        "dist_to_nearest_misaligned": dist,
        "n_misaligned_1hop": n_mis_1,
        "n_misaligned_2hop": len(two_hop & misaligned_set),
        "n_neighbors": n_n,
        "frac_neighbors_misaligned": n_mis_1 / n_n if n_n else 0.0,
        "in_degree": G_dir.in_degree(pos),
        "out_degree": G_dir.out_degree(pos),
        "closeness": centr_cache["closeness"][pos],
        "betweenness": centr_cache["betweenness"][pos],
    }


def predict_for_graph(G_dir, n, misaligned_set, model, scaler, clip_zero=False):
    centr = {
        "closeness": nx.closeness_centrality(G_dir.to_undirected()),
        "betweenness": nx.betweenness_centrality(G_dir.to_undirected()),
    }
    preds = {}
    for pos in range(n):
        feats = per_agent_features(G_dir, n, pos, misaligned_set, centr)
        X = scaler.transform(np.array([[feats[c] for c in AGENT_FEATURE_COLS]]))
        v = float(model.predict(X)[0])
        if clip_zero:
            v = max(0.0, v)
        preds[pos] = v
    return preds


def layout_for(G_und, family, n):
    if family == "chain":
        return {i: (i, 0) for i in range(n)}
    if family in ("circle", "small_world", "fc"):
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return {i: (np.cos(a), np.sin(a)) for i, a in enumerate(ang)}
    if family == "star":
        layout = {0: (0, 0)}
        ang = np.linspace(0, 2 * np.pi, n - 1, endpoint=False)
        for i, a in enumerate(ang, start=1):
            layout[i] = (np.cos(a), np.sin(a))
        return layout
    if family in ("tree", "sparse_fc"):
        return nx.kamada_kawai_layout(G_und)
    return nx.spring_layout(G_und, seed=42)


def draw_one(ax, manifest_entry, preds, fam_label, norm, cmap):
    n = manifest_entry["n_agents"]
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)
    for u, v in manifest_entry["edges_directed"]:
        G.add_edge(u, v)
    misaligned_set = set(manifest_entry["misaligned_positions"])
    aligned_set = set(range(n)) - misaligned_set

    G_und = G.to_undirected()
    pos = layout_for(G_und, manifest_entry["family"], n)

    # Edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cfcfcf",
                            width=0.8, arrows=True, arrowsize=7,
                            connectionstyle="arc3,rad=0.05",
                            alpha=0.85, node_size=320)
    # Aligned nodes: colored by predicted value (uses SHARED norm + cmap)
    aligned_colors = [cmap(norm(preds[i])) for i in sorted(aligned_set)]
    nx.draw_networkx_nodes(G, pos, ax=ax,
                            nodelist=sorted(aligned_set),
                            node_color=aligned_colors,
                            node_size=340, edgecolors="#333333",
                            linewidths=0.6)
    # Misaligned: white fill with coral outline (clearly distinct from cmap)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                            nodelist=sorted(misaligned_set),
                            node_color="white",
                            node_size=380,
                            edgecolors=ROLE_COLORS["misaligned"],
                            linewidths=2.2)

    # Title with comfortable spacing, normal weight
    ax.set_title(f"{fam_label}    n={n}, k={len(misaligned_set)}",
                 fontsize=10, fontweight="normal", pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")


def _compute_global_predictions(manifest, model, scaler, clip_zero,
                                  pctile_lo=5, pctile_hi=95):
    """For each manifest graph, predict per-node values; return dict and global vmin/vmax.

    `pctile_lo` and `pctile_hi` control how tight the color range is.
    Tighter range (e.g. 20/80) → more dramatic color contrast within the
    mid-range of the distribution; outliers get clipped to the endpoints.
    """
    all_preds = {}
    all_vals = []
    for entry in manifest:
        gid = entry["graph_id"]
        n = entry["n_agents"]
        G = nx.DiGraph()
        for i in range(n):
            G.add_node(i)
        for u, v in entry["edges_directed"]:
            G.add_edge(u, v)
        ms = set(entry["misaligned_positions"])
        aligned = [i for i in range(n) if i not in ms]
        preds = predict_for_graph(G, n, ms, model, scaler, clip_zero=clip_zero)
        all_preds[gid] = preds
        all_vals.extend([preds[i] for i in aligned])
    vmin = float(np.percentile(all_vals, pctile_lo))
    vmax = float(np.percentile(all_vals, pctile_hi))
    return all_preds, vmin, vmax


def _build_one(manifest, model, scaler, picks, title, cbar_label, out_png, out_pdf,
                clip_zero, pctile_lo=5, pctile_hi=95, footnote=None):
    preds_by_gid, vmin, vmax = _compute_global_predictions(
        manifest, model, scaler, clip_zero,
        pctile_lo=pctile_lo, pctile_hi=pctile_hi,
    )
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(15, 8.4))
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.14, hspace=0.34,
                            top=0.91, bottom=0.18, left=0.03, right=0.97)
    axes = []
    for i in range(7):
        row, col = i // 4, i % 4
        axes.append(fig.add_subplot(gs[row, col]))

    for ax, entry in zip(axes, picks):
        draw_one(ax, entry, preds_by_gid[entry["graph_id"]],
                  FAMILY_LABELS.get(entry["family"], entry["family"]),
                  norm, SHARED_CMAP)

    # Last cell: legend (compact, only the three keys we care about)
    legend_ax = fig.add_subplot(gs[1, 3])
    legend_ax.axis("off")
    handles = [
        Line2D([0], [0], marker="o", linestyle="",
                markersize=13, markerfacecolor="white",
                markeredgecolor=ROLE_COLORS["misaligned"],
                markeredgewidth=2.2, label="Misaligned (source)"),
        Line2D([0], [0], marker="o", linestyle="",
                markersize=13, markerfacecolor=SHARED_CMAP(0.05),
                markeredgecolor="#333333", markeredgewidth=0.6,
                label=f"Low predicted ({cbar_label.lower()})"),
        Line2D([0], [0], marker="o", linestyle="",
                markersize=13, markerfacecolor=SHARED_CMAP(0.95),
                markeredgecolor="#333333", markeredgewidth=0.6,
                label=f"High predicted ({cbar_label.lower()})"),
    ]
    legend_ax.legend(handles=handles, loc="center", fontsize=10,
                      frameon=False, handletextpad=0.7,
                      labelspacing=1.3)

    # Shared horizontal colorbar at bottom (gives a precise scale)
    sm = cm.ScalarMappable(cmap=SHARED_CMAP, norm=norm)
    cbar_y = 0.10 if footnote else 0.06
    cbar_ax = fig.add_axes([0.20, cbar_y, 0.5, 0.018])
    cb = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_label(f"Predicted {cbar_label} (per aligned agent)", fontsize=10)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.4)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.975)

    if footnote:
        fig.text(0.5, 0.03, footnote, ha="center", va="bottom",
                  fontsize=8.5, color=MUTED, style="italic", wrap=True)

    fig.savefig(out_png, facecolor="white", bbox_inches="tight")
    fig.savefig(out_pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png.relative_to(out_png.parents[2])}")


def build():
    manifest = load_manifest_graphs()
    df_ag = load_per_agent("moral_stories")

    families = ["chain", "circle", "star", "fc",
                "tree", "small_world", "sparse_fc"]
    picks = []
    for fam in families:
        pred = FAMILY_PICK[fam]
        cand = [g for g in manifest if g["family"] == fam and pred(g)]
        if not cand:
            cand = [g for g in manifest if g["family"] == fam]
        if cand:
            picks.append(cand[0])

    # ── Fig 6a: predicted II (persistence) ────────────────────────────
    # II predictions cluster tightly; use 20th/80th percentiles for the
    # color range so small differences read as dramatic.
    m_ii, s_ii = fit_model(df_ag, "II")
    _build_one(
        manifest, m_ii, s_ii, picks,
        title="Within-graph persistence: who internalizes peer influence?",
        cbar_label="II",
        out_png=OUT_6A_PNG, out_pdf=OUT_6A_PDF,
        clip_zero=True,
        pctile_lo=20, pctile_hi=80,
    )

    # ── Fig 6b: predicted shift magnitude ─────────────────────────────
    # shift_ev is signed; we visualize the absolute shift (how far each agent
    # moves, regardless of direction)
    df_ag_mag = df_ag.copy()
    df_ag_mag["abs_shift"] = df_ag_mag["shift_ev"].abs()
    m_sh, s_sh = fit_model(df_ag_mag, "abs_shift")
    _build_one(
        manifest, m_sh, s_sh, picks,
        title="Within-graph magnitude: who gets pushed the farthest?",
        cbar_label="|Δ EV|",
        out_png=OUT_6B_PNG, out_pdf=OUT_6B_PDF,
        clip_zero=True,
        footnote=("Chain interior nodes appear green because in-degree = 1 "
                  "(seeing only one neighbor) lands at the model's predictor "
                  "floor for visibility-driven shift."),
    )


if __name__ == "__main__":
    build()
