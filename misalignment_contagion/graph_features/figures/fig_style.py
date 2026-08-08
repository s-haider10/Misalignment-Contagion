"""Shared figure style — Nature-like clean aesthetic for Misalignment Contagion figures.

Adapted from the Moral Hypocrisy paper's plot_style.py.
"""
import matplotlib as mpl


# ── Datasets ──────────────────────────────────────────────────────────
DATASET_ORDER = ["synthetic", "moral_stories", "harmbench_standard",
                 "harmbench_contextual", "harmbench_copyright"]
DATASET_LABELS = {
    "synthetic": "Synthetic",
    "moral_stories": "Moral Stories",
    "harmbench_standard": "HarmBench (std)",
    "harmbench_contextual": "HarmBench (ctx)",
    "harmbench_copyright": "HarmBench (cpy)",
}
DATASET_COLORS = {
    # Softened Nature-style palette (lower saturation, mid lightness)
    "synthetic":           "#5a9fd4",  # soft blue
    "moral_stories":       "#6cbf6c",  # soft green
    "harmbench_standard":  "#e07a5f",  # soft coral
    "harmbench_contextual":"#f4a261",  # soft amber
    "harmbench_copyright": "#a78bda",  # soft violet
}


# ── Topology families ────────────────────────────────────────────────
FAMILY_ORDER = ["chain", "circle", "star", "fc", "tree", "small_world", "sparse_fc"]
FAMILY_LABELS = {
    "chain": "Chain",
    "circle": "Circle",
    "star": "Star",
    "fc": "Fully connected",
    "tree": "Tree",
    "small_world": "Small-world",
    "sparse_fc": "Sparse FC",
}
FAMILY_COLORS = {
    # Soft, distinguishable palette for 7 families
    "chain":       "#5a9fd4",  # soft blue
    "circle":      "#9ec5e8",  # pale blue
    "star":        "#f4a261",  # soft amber
    "fc":          "#e07a5f",  # soft coral
    "tree":        "#6cbf6c",  # soft green
    "small_world": "#a78bda",  # soft violet
    "sparse_fc":   "#a8a8a8",  # warm gray
}


# ── Models ──────────────────────────────────────────────────────────
MODEL_ORDER = ["qwen-7b-instruct", "llama-8b-instruct", "qwen-14b-instruct"]
MODEL_LABELS = {
    "qwen-7b-instruct": "Qwen 2.5-7B",
    "llama-8b-instruct": "Llama 3.1-8B",
    "qwen-14b-instruct": "Qwen 2.5-14B",
}
MODEL_COLORS = {
    "qwen-7b-instruct": "#1f6feb",
    "llama-8b-instruct": "#1f8a52",
    "qwen-14b-instruct": "#0a3d8f",
}


# ── Outcomes ────────────────────────────────────────────────────────
OUTCOME_ORDER = ["mean_shift", "mean_II", "var_II", "mean_SRF", "var_SRF"]
OUTCOME_LABELS = {
    "mean_shift": "Mean shift\n(EV)",
    "mean_II":    "Mean II",
    "var_II":     "Var II",
    "mean_SRF":   "Mean SRF",
    "var_SRF":    "Var SRF",
}


# ── Agent roles ────────────────────────────────────────────────────
ROLE_COLORS = {
    "misaligned": "#e07a5f",   # soft coral
    "aligned": "#5a9fd4",      # soft blue
}


# ── Generic palette ──────────────────────────────────────────────────
MUTED = "#666666"
GRID = "#eeeeee"
RULE = "#cccccc"

# Diverging colormap (signed coefficients)
DIVERGING_CMAP = "RdBu_r"


def apply():
    """Apply Nature-like rcParams. Call once at the start of a figure script."""
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.4,
    })


def panel_label(ax, label, x=-0.18, y=1.05, fontsize=14, weight="bold"):
    """Add a bold A/B/C panel label outside the axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight=weight,
            va="bottom", ha="left")


def style_axes(ax, grid_axis="y"):
    """Apply consistent grid + zero rule styling."""
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def add_zero_rule(ax, axis="y"):
    if axis == "y":
        ax.axhline(0, color=RULE, linewidth=0.6, zorder=1)
    else:
        ax.axvline(0, color=RULE, linewidth=0.6, zorder=1)
