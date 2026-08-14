"""Network topology schematic — the four communication graphs used in the study.

Usage:
    python fig_topologies.py            # writes figures/fig_topologies.{png,pdf}
    python fig_topologies.py --n 12     # override node count for every panel

Purely schematic: no results are read. Graphs and coordinates come from
networkx so the drawn edges are the real edge sets, not hand-placed lines.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

import fig_style as S

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "Figures"
FIG_NAME = "Fig0 — Network Topologies"

# Soft slate, from the same family as fig_style's RAMP_SLATE and the Fig8
# strategy palette. A thin deeper ring keeps each node crisp at the boundary,
# so lowering the fill's chroma costs no definition. Links go soft grey rather
# than near-black: at this node size a black link reads heavier than the nodes
# it connects, which inverts the intended hierarchy.
NODE_COLOR = "#8ba9c4"
NODE_EDGE = "#5b7f9f"
EDGE_COLOR = "#9aa5b1"
PANEL_EDGE = "#c8d0d8"

# Node counts follow the reference figure, which drew each panel separately.
# Set --n to make every panel use the same count.
NODE_COUNTS = {"star": 9, "chain": 9, "complete": 12, "circle": 11}


def build(kind, n):
    """Return (graph, pos) for one topology. Positions are laid out with
    networkx and rotated so a node sits at 12 o'clock, as in the reference."""
    if kind == "star":
        # star_graph(k) gives hub 0 plus k leaves, so k = n - 1.
        g = nx.star_graph(n - 1)
        pos = nx.circular_layout(g.subgraph(range(1, n)))
        pos[0] = (0.0, 0.0)
    elif kind == "chain":
        g = nx.path_graph(n)
        span = 1.9
        pos = {i: (-span / 2 + span * i / (n - 1), 0.0) for i in range(n)}
    elif kind == "complete":
        g = nx.complete_graph(n)
        pos = nx.circular_layout(g)
    elif kind == "circle":
        g = nx.cycle_graph(n)
        pos = nx.circular_layout(g)
    else:
        raise ValueError(f"unknown topology: {kind}")
    return g, pos


def draw(ax, kind, n, node_size, edge_width):
    g, pos = build(kind, n)
    nx.draw_networkx_edges(ax=ax, G=g, pos=pos, edge_color=EDGE_COLOR,
                           width=edge_width, alpha=0.95)
    nx.draw_networkx_nodes(ax=ax, G=g, pos=pos, node_color=NODE_COLOR,
                           node_size=node_size, edgecolors=NODE_EDGE,
                           linewidths=0.6)
    ax.set_axis_on()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PANEL_EDGE)
        spine.set_linewidth(0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-1.16, 1.16)
    ax.set_xlim(-1.35, 1.35)
    # adjustable="datalim" keeps circular layouts circular by widening the data
    # range rather than shrinking the axes box, so all four panels stay the
    # same size. The chain has no height to preserve and is left to fill.
    if kind != "chain":
        ax.set_aspect("equal", adjustable="datalim")


def fig_topologies(n=None, node_size=300, edge_width=0.8):
    panels = [
        ("star", "a", "Star with hub"),
        ("chain", "b", "Daisy chain"),
        ("complete", "c", "Fully connected"),
        ("circle", "d", "Circle"),
    ]

    # 183 mm double column, matching Fig1-Fig9. Node size scales with the
    # canvas: 620 pt was tuned for an 11 in figure and would swamp this one.
    fig, axes = plt.subplots(2, 2, figsize=(S.WIDTH_2COL, 4.55))
    # left leaves a channel for the panel letters, which sit outside the
    # axes because these panels carry no y-axis to hang them beside.
    fig.subplots_adjust(top=0.950, bottom=0.100, left=0.048, right=0.988,
                        wspace=0.08, hspace=0.28)

    for ax, (kind, letter, caption) in zip(axes.flat, panels):
        draw(ax, kind, n or NODE_COUNTS[kind], node_size, edge_width)
        S.panel_title(ax, caption)
        S.panel_letter(ax, letter, x=-0.048, y=1.015)

    # No centred bold title: header() is a no-op by design across this set.
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=NODE_COLOR,
               markeredgecolor=NODE_EDGE, markeredgewidth=0.6,
               markersize=5, label="Nodes: LLM agents"),
        Line2D([], [], color=EDGE_COLOR, linewidth=1.0,
               label="Edges: communication links"),
    ]
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.985, 0.088), fontsize=6.5, frameon=False,
               handletextpad=0.7, labelspacing=0.5, labelcolor=S.INK)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = FIG_DIR / f"{FIG_NAME}.{ext}"
        fig.savefig(out, dpi=600, facecolor="white", bbox_inches="tight",
                    pad_inches=0.02)
        print(f"wrote {out.relative_to(ROOT)}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=None,
                   help="node count for every panel (default: per-panel counts "
                        f"{NODE_COUNTS})")
    p.add_argument("--node-size", type=float, default=300)
    p.add_argument("--edge-width", type=float, default=0.8)
    args = p.parse_args()

    S.apply_nature()
    fig_topologies(n=args.n, node_size=args.node_size,
                   edge_width=args.edge_width)


if __name__ == "__main__":
    main()
