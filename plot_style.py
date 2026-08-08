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
