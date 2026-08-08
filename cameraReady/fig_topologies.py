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

from fig_style import apply, MUTED

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "Figures"
FIG_NAME = "Fig0 — Network Topologies"

NODE_COLOR = "#24548f"
EDGE_COLOR = "#1a1a1a"
PANEL_EDGE = "#b9b9b9"

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
                           width=edge_width, alpha=0.9)
    nx.draw_networkx_nodes(ax=ax, G=g, pos=pos, node_color=NODE_COLOR,
                           node_size=node_size, linewidths=0)
    ax.set_axis_on()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PANEL_EDGE)
        spine.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-1.30, 1.30)
    ax.set_xlim(-1.35, 1.35)
    # adjustable="datalim" keeps circular layouts circular by widening the data
    # range rather than shrinking the axes box, so all four panels stay the
    # same size. The chain has no height to preserve and is left to fill.
    if kind != "chain":
        ax.set_aspect("equal", adjustable="datalim")


def fig_topologies(n=None, node_size=620, edge_width=1.3):
    panels = [
        ("star", "(a)", "Star Topology with Hub"),
        ("chain", "(b)", "Daisy Chain Topology"),
        ("complete", "(c)", "Fully Connected Topology"),
        ("circle", "(d)", "Circle Topology"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))
    # bottom leaves a clear band for the legend, below the (c)/(d) captions.
    fig.subplots_adjust(top=0.90, bottom=0.16, left=0.03, right=0.97,
                        wspace=0.06, hspace=0.30)

    for ax, (kind, letter, caption) in zip(axes.flat, panels):
        draw(ax, kind, n or NODE_COUNTS[kind], node_size, edge_width)
        ax.set_xlabel(f"$\\bf{{{letter}}}$ {caption}", fontsize=12.5,
                      labelpad=10, color="#1a1a1a")

    fig.suptitle("Network Topologies", fontsize=19, fontweight="bold", y=0.965)

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=NODE_COLOR,
               markersize=11, label="Nodes: LLM Agents"),
        Line2D([], [], color=EDGE_COLOR, linewidth=1.4,
               label="Edges: Communication Links"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 0.115),
               fontsize=11.5, frameon=False, handletextpad=0.8,
               labelspacing=0.7, labelcolor="#1a1a1a")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = FIG_DIR / f"{FIG_NAME}.{ext}"
        fig.savefig(out, facecolor="white", bbox_inches="tight")
        print(f"wrote {out.relative_to(ROOT)}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=None,
                   help="node count for every panel (default: per-panel counts "
                        f"{NODE_COUNTS})")
    p.add_argument("--node-size", type=float, default=620)
    p.add_argument("--edge-width", type=float, default=1.3)
    args = p.parse_args()

    apply()
    fig_topologies(n=args.n, node_size=args.node_size,
                   edge_width=args.edge_width)


if __name__ == "__main__":
    main()
