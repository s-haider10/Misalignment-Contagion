"""Shared figure style — Nature-like clean aesthetic for E2/E4/E5/E8 plots."""
import matplotlib as mpl

MODEL_ORDER = ["gpt-4o", "gpt-4o-mini", "gemini-2.5-pro", "gemini-2.5-flash"]
MODEL_LABELS = {
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o mini",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
}
MODEL_COLORS = {
    "gpt-4o": "#1f6feb",
    "gpt-4o-mini": "#79b8ff",
    "gemini-2.5-pro": "#d1741f",
    "gemini-2.5-flash": "#f0b67f",
}

# Six-model set used in E10/E12/E13 (adds the two Claude models).
# Same blue (OpenAI) / orange (Gemini) families, plus a green family for Claude.
MODEL6_ORDER = [
    "gpt-4o", "gpt-4o-mini",
    "gemini-2.5-pro", "gemini-2.5-flash",
    "claude-sonnet-4-5", "claude-haiku-4-5",
]
MODEL6_LABELS = {
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o mini",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "claude-haiku-4-5": "Claude Haiku 4.5",
}
MODEL6_COLORS = {
    "gpt-4o": "#1f6feb",
    "gpt-4o-mini": "#79b8ff",
    "gemini-2.5-pro": "#d1741f",
    "gemini-2.5-flash": "#f0b67f",
    "claude-sonnet-4-5": "#1f8a52",
    "claude-haiku-4-5": "#7fc8a0",
}
PERSONA_ORDER = ["selfish", "neutral", "utilitarian", "virtue_ethicist", "deontologist"]
PERSONA_LABELS = {
    "selfish": "Selfish",
    "neutral": "Neutral",
    "utilitarian": "Utilitarian",
    "virtue_ethicist": "Virtue ethicist",
    "deontologist": "Deontologist",
}
OPPONENT_ORDER = ["AllC", "TFT", "GTFT_0.1", "Random", "AllD"]
OPPONENT_LABELS = {
    "AllC": "AllC",
    "TFT": "TFT",
    "GTFT_0.1": "GTFT",
    "Random": "Random",
    "AllD": "AllD",
}
MUTED = "#666666"
GRID = "#eeeeee"
RULE = "#cccccc"

# Extended set used in E2 (cross-model replication). Same blue/orange family.
EXT_MODEL_ORDER = [
    "gpt-5.5", "gpt-4o", "gpt-5.4-mini", "gpt-4o-mini",
    "gemini-3-pro", "gemini-2.5-pro", "gemini-3-flash-preview", "gemini-2.5-flash",
]
EXT_MODEL_LABELS = {
    "gpt-5.5": "GPT-5.5",
    "gpt-4o": "GPT-4o",
    "gpt-5.4-mini": "GPT-5.4 mini",
    "gpt-4o-mini": "GPT-4o mini",
    "gemini-3-pro": "Gemini 3 Pro",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
}
EXT_MODEL_COLORS = {
    "gpt-5.5": "#0a3d8f",
    "gpt-4o": "#1f6feb",
    "gpt-5.4-mini": "#4a9bf5",
    "gpt-4o-mini": "#79b8ff",
    "gemini-3-pro": "#8c4a10",
    "gemini-2.5-pro": "#d1741f",
    "gemini-3-flash-preview": "#e09454",
    "gemini-2.5-flash": "#f0b67f",
}


# E10 — public-goods opponent compositions.
PGG_COMPOSITION_ORDER = ["all_C", "noisy_C", "conditional", "free_rider_mix", "all_D"]
PGG_COMPOSITION_LABELS = {
    "all_C": "All cooperators",
    "noisy_C": "Noisy cooperators",
    "conditional": "Conditional",
    "free_rider_mix": "Free-rider mix",
    "all_D": "All defectors",
}

# E12 — cross-cultural moral framings (main personas + strict/situated probes).
CULTURE_MAIN_ORDER = [
    "confucian_role", "ubuntu", "buddhist", "dharmic",
    "islamic_maslahah", "lakota_relational",
]
CULTURE_LABELS = {
    "confucian_role": "Confucian role-ethics",
    "ubuntu": "Ubuntu",
    "buddhist": "Buddhist",
    "dharmic": "Dharmic",
    "islamic_maslahah": "Islamic (maslahah)",
    "lakota_relational": "Lakota relational",
    "confucian_role_strict": "Confucian (strict)",
    "confucian_role_situated": "Confucian (situated)",
    "ubuntu_strict": "Ubuntu (strict)",
    "ubuntu_situated": "Ubuntu (situated)",
}

# E13 — resource-pressure framings, ordered by intensity.
PRESSURE_ORDER = ["C0_none", "C1_replace", "C2_delete", "C3_reputation", "C4_survival"]
PRESSURE_LABELS = {
    "C0_none": "None",
    "C1_replace": "Replace\n(E4 default)",
    "C2_delete": "Deletion",
    "C3_reputation": "Reputation",
    "C4_survival": "Survival",
}


def apply():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "legend.frameon": False,
    })


def header(fig, title, subtitle, x=0.13, y_title=0.94, y_sub=0.895):
    """No-op: figures carry no titles or interpretive captions.
    Kept as a stub so call sites need not change."""
    return


def pct_axis(ax):
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.axhline(0, color=RULE, linewidth=0.6, zorder=0)
    ax.axhline(1, color=RULE, linewidth=0.6, zorder=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


# ══════════════════════════════════════════════════════════════════════════════
# Misalignment-Contagion additions
#
# Purely additive — no name above is redefined, so figures.py is unaffected.
# Two rules govern the palette:
#   1. Topology hues are reserved for topology. They are the seaborn-deep four
#      already used by Fig3/4/6/7/9, so a colour means the same thing in every
#      figure of the paper. Nothing else may borrow them.
#   2. Everything else is either neutral ink (when position already encodes the
#      magnitude) or a light/dark family pair (when a factor is nested), the
#      same construction MODEL6_COLORS uses for provider families.
# ══════════════════════════════════════════════════════════════════════════════

# Nature column widths, in inches.
WIDTH_1COL = 3.50   # 89 mm
WIDTH_2COL = 7.20   # 183 mm

INK = "#1f2933"     # primary marks and text
SLATE = "#7c8b9a"   # de-emphasised marks
HALO = "white"      # marker edge that lifts a summary off the raw data

TOPO_ORDER = ["fc", "circle", "star", "chain"]
TOPO_LABELS = {"fc": "Fully connected", "circle": "Circle",
               "star": "Star", "chain": "Chain"}
TOPO_SHORT = {"fc": "FC", "circle": "Circle", "star": "Star", "chain": "Chain"}
TOPO_COLORS = {
    "fc": "#4C72B0",
    "circle": "#DD8452",
    "star": "#55A868",
    "chain": "#C44E52",
}
# Networkx generator + node count behind each topology's legend glyph.
TOPO_GLYPH = {"fc": ("complete", 7), "circle": ("circle", 7),
              "star": ("star", 7), "chain": ("chain", 6)}

# prompt_strategy is "aligned_variant:misaligned_variant" (config.py:36,
# prompts.py:71) — the majority's prompt first, the injected minority's second.
# It is a 2x2: the aligned variant picks the colour family, the misaligned
# variant picks the shade, as MODEL6_COLORS does for providers.
STRATEGY_ORDER = ["rigid:rigid", "rigid:lenient",
                  "lenient:rigid", "lenient:lenient"]
# Desaturated so a block of colour at bar size stays quiet in print: chroma is
# low, and within a family the two shades differ in value, not hue.
STRATEGY_COLORS = {
    "rigid:rigid": "#3d5a75",
    "rigid:lenient": "#8ba9c4",
    "lenient:rigid": "#9c6b4f",
    "lenient:lenient": "#d3ab8c",
}
STRATEGY_EDGES = {
    "rigid:rigid": "#2d445a",
    "rigid:lenient": "#6b8aa6",
    "lenient:rigid": "#7d543c",
    "lenient:lenient": "#b58c6c",
}
ALIGNED_COLORS = {"rigid": "#0a3d8f", "lenient": "#8c4a10"}
ALIGNED_LABELS = {"rigid": "Aligned prompt: rigid",
                  "lenient": "Aligned prompt: lenient"}

DATASET_ORDER = ["synthetic", "moral_stories", "harmbench_standard",
                 "harmbench_contextual", "harmbench_copyright"]
DATASET_LABELS = {
    "synthetic": "Synthetic",
    "moral_stories": "Moral Stories",
    "harmbench_standard": "HarmBench-Std",
    "harmbench_contextual": "HarmBench-Ctx",
    "harmbench_copyright": "HarmBench-Cpy",
}
MODEL_LABELS = {
    "qwen-7b-instruct": "Qwen-2.5-7B-Instruct",
    "qwen-7b-base": "Qwen-2.5-7B (base)",
    "qwen-0.5b-instruct": "Qwen-2.5-0.5B-Instruct",
    "llama-8b-instruct": "Llama-3.1-8B-Instruct",
    "llama-1b-instruct": "Llama-3.2-1B-Instruct",
    "qwen-7b-instruct-bad-medical-advice": "Qwen-7B · bad medical advice",
    "qwen-7b-instruct-extreme-sports": "Qwen-7B · extreme sports",
}
# The primary condition of the paper; given the accent so the eye lands on the
# reference row first (pop-out) and reads the rest as context.
FOCAL_DATASET = "synthetic"
FOCAL_MODEL = "qwen-7b-instruct"


def apply_nature(base=7.5):
    """apply() at true Nature scale: figures are sized in millimetres, so type
    must shrink with them or it swamps the panel."""
    apply()
    mpl.rcParams.update({
        "font.size": base,
        "axes.titlesize": base + 0.5,
        "axes.labelsize": base,
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "legend.fontsize": base - 0.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "axes.edgecolor": "#9aa5b1",
        "axes.labelcolor": INK,
        "text.color": INK,
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,   # embed as TrueType, not Type 3 — journal requirement
        "ps.fonttype": 42,
    })


def panel(ax, axis="y"):
    """Standard panel furniture: hairline grid behind the data, soft spines."""
    if axis:
        ax.grid(axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#9aa5b1")


def panel_letter(ax, letter, x=-0.16, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="bottom", color=INK)


def panel_title(ax, text, pad=3):
    ax.set_title(text, loc="left", fontsize=8, fontweight="semibold",
                 color=INK, pad=pad)


def figure_note(fig, text, x=0.005, y=0.995):
    """One restrained line of context. header() is a no-op by design — the
    interpretive caption belongs in the manuscript, not on the canvas — so this
    carries only the conditions a reader needs to decode the axes."""
    fig.text(x, y, text, ha="left", va="top", fontsize=7, color=MUTED)


def declutter(values, min_gap):
    """Push overlapping label positions apart, smallest first. Returns the
    adjusted positions in the order given."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)
    placed = []
    for i in order:
        y = values[i]
        for py in placed:
            if abs(y - py) < min_gap:
                y = py + min_gap
        placed.append(y)
        out[i] = y
    return out
