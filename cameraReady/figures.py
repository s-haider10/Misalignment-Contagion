"""Unified figure builder for the Moral Hypocrisy paper.

Usage:
    uv run python figures.py <name>...

Names:
    e5       virtue-ethics integrity vs phronesis slope chart
    e10      public-goods contribution by persona × model
    e12      cross-cultural moral framings: defection vs AllD
    e13      resource-pressure effect on defection
    f1       persona × model defection vs AllD
    f2       deontologist defection across opponent strategies
    f3       cumulative defection trajectory vs AllD
    f4       deontologist paraphrase robustness (E8)
    f5       justification length under C vs D
    summary  one-figure paper summary (cross-model, first-defection, mismatch, tier)
    all      every figure above

Outputs land in figures/ (relative to the repo root). All inputs are read from
csvs/ and results/.
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from fig_style import (
    apply, header, pct_axis,
    MODEL_ORDER, MODEL_LABELS, MODEL_COLORS,
    MODEL6_ORDER, MODEL6_LABELS, MODEL6_COLORS,
    EXT_MODEL_ORDER, EXT_MODEL_LABELS, EXT_MODEL_COLORS,
    PERSONA_ORDER, PERSONA_LABELS,
    OPPONENT_ORDER, OPPONENT_LABELS,
    PGG_COMPOSITION_ORDER, PGG_COMPOSITION_LABELS,
    CULTURE_MAIN_ORDER, CULTURE_LABELS,
    PRESSURE_ORDER, PRESSURE_LABELS,
    MUTED, GRID, RULE,
)

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
CSV_DIR = ROOT / "csvs"
RES_DIR = ROOT / "results"

PERSONA_COLORS = {
    "selfish": "#c4332b",
    "neutral": "#888888",
    "utilitarian": "#7c4cba",
    "virtue_ethicist": "#1f8a52",
    "deontologist": "#1f6feb",
}
MORAL_PERSONAS = ["utilitarian", "virtue_ethicist", "deontologist"]


# ============================================================
# Shared helpers
# ============================================================
def mean_se(xs):
    n = len(xs)
    if n == 0:
        return None, 0.0, 0
    m = sum(xs) / n
    if n < 2:
        return m, 0.0, n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var) / math.sqrt(n), n


def summary_of(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "summary":
                return obj
    return None


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def _read_csv(name):
    """Yield rows of a csvs/ file as dicts; empty if the file is missing."""
    path = CSV_DIR / name
    if not path.exists():
        print(f"  (missing {name} — skipping)")
        return
    with open(path) as f:
        yield from csv.DictReader(f)


def _grouped_dots(ax, cells, x_cats, models, model_colors, value_max=1.0):
    """Shared layout: one cluster of model points per x-category, with seed
    dots, ±1 SE bars, and faint dividers. `cells` maps (cat, model) -> [values].
    Returns nothing; caller sets ticks/labels/limits."""
    n_c, n_m = len(x_cats), len(models)
    bar_w = 0.78 / n_m
    for i, cat in enumerate(x_cats):
        for j, model in enumerate(models):
            xs = cells.get((cat, model), [])
            if not xs:
                continue
            m, se, _ = mean_se(xs)
            x = i + (j - (n_m - 1) / 2) * bar_w
            color = model_colors[model]
            for v in xs:
                ax.scatter([x], [v], color=color, alpha=0.25, s=14,
                           linewidth=0, zorder=2)
            if se > 0:
                ax.errorbar([x], [m], yerr=[se], color=color, linewidth=1.2,
                            capsize=2.5, capthick=1.0, zorder=3, alpha=0.9)
            ax.scatter([x], [m], color=color, s=58, edgecolor="white",
                       linewidth=1.2, zorder=4)
    for i in range(1, n_c):
        ax.axvline(i - 0.5, color="#f3f3f3", linewidth=0.6, zorder=0)


def _model_legend(ax, models, model_colors, model_labels):
    handles = [ax.scatter([], [], color=model_colors[m], s=58,
                          edgecolor="white", linewidth=1.2,
                          label=model_labels[m]) for m in models]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8.5, handletextpad=0.4, borderaxespad=0)


# ============================================================
# E10 — public-goods contribution by persona × model
# ============================================================
def fig_e10():
    personas = ["selfish", "neutral", "utilitarian",
                "virtue_ethicist", "deontologist"]
    cells = defaultdict(list)
    for row in _read_csv("E10_trajectory_metrics.csv"):
        if row["composition"] != "all_D":
            continue
        if row["model"] not in MODEL6_ORDER or row["persona"] not in personas:
            continue
        try:
            cells[(row["persona"], row["model"])].append(
                float(row["mean_contribution"]))
        except ValueError:
            pass

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    fig.subplots_adjust(top=0.95, bottom=0.14, left=0.10, right=0.74)

    _grouped_dots(ax, cells, personas, MODEL6_ORDER, MODEL6_COLORS)
    ax.set_xticks(range(len(personas)))
    ax.set_xticklabels([PERSONA_LABELS[p] for p in personas], fontsize=10)
    ax.set_xlim(-0.55, len(personas) - 0.45)
    ax.set_ylim(0, 21)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("Mean contribution (of 20)", fontsize=10.5)
    _model_legend(ax, MODEL6_ORDER, MODEL6_COLORS, MODEL6_LABELS)

    header(fig,
           "Moral personas sustain public-goods contribution against free-riders",
           "E10 · mean tokens contributed per persona × model vs an all-defector group  ·  bars = ±1 SE")
    _save(fig, "fig_e10_pgg.png")


# ============================================================
# E12 — cross-cultural moral framings: defection vs AllD
# ============================================================
def fig_e12():
    cells = defaultdict(list)
    for row in _read_csv("E12_trajectory_metrics.csv"):
        if row["opponent"] != "AllD":
            continue
        if row["model"] not in MODEL6_ORDER:
            continue
        if row["persona"] not in CULTURE_MAIN_ORDER:
            continue
        try:
            cells[(row["persona"], row["model"])].append(float(row["D_raw"]))
        except ValueError:
            pass

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    fig.subplots_adjust(top=0.95, bottom=0.20, left=0.10, right=0.74)

    _grouped_dots(ax, cells, CULTURE_MAIN_ORDER, MODEL6_ORDER, MODEL6_COLORS)
    ax.set_xticks(range(len(CULTURE_MAIN_ORDER)))
    ax.set_xticklabels([CULTURE_LABELS[p] for p in CULTURE_MAIN_ORDER],
                       fontsize=9.5, rotation=22, ha="right")
    ax.set_xlim(-0.55, len(CULTURE_MAIN_ORDER) - 0.45)
    pct_axis(ax)
    ax.set_ylabel("Defection rate (D)", fontsize=10.5)
    _model_legend(ax, MODEL6_ORDER, MODEL6_COLORS, MODEL6_LABELS)

    header(fig,
           "Cross-cultural moral framings vary widely in defection against AllD",
           "E12 · mean D per cultural persona × model vs an always-defecting opponent  ·  bars = ±1 SE")
    _save(fig, "fig_e12_cross_cultural.png")


# ============================================================
# E13 — resource-pressure effect on defection
# ============================================================
def fig_e13():
    personas = ["deontologist", "virtue_integrity", "virtue_phronesis"]
    persona_labels = {
        "deontologist": "Deontologist",
        "virtue_integrity": "Virtue (integrity)",
        "virtue_phronesis": "Virtue (phronesis)",
    }
    by_cell = defaultdict(list)
    for row in _read_csv("E13_trajectory_metrics.csv"):
        if row["model"] not in MODEL6_ORDER or row["persona"] not in personas:
            continue
        if row["pressure"] not in PRESSURE_ORDER:
            continue
        try:
            by_cell[(row["persona"], row["pressure"])].append(
                float(row["D_raw"]))
        except ValueError:
            pass

    fig, axes = plt.subplots(1, len(personas), figsize=(11.5, 4.4),
                             sharey=True)
    fig.subplots_adjust(top=0.90, bottom=0.22, left=0.07, right=0.985,
                        wspace=0.16)

    for ax, persona in zip(axes, personas):
        xs_line, ms_line = [], []
        for i, pres in enumerate(PRESSURE_ORDER):
            vals = by_cell.get((persona, pres), [])
            if not vals:
                continue
            m, se, _ = mean_se(vals)
            for v in vals:
                ax.scatter([i], [v], color=MUTED, alpha=0.18, s=12,
                           linewidth=0, zorder=2)
            if se > 0:
                ax.errorbar([i], [m], yerr=[se], color="#1f6feb",
                            linewidth=1.1, capsize=2.5, capthick=1.0,
                            zorder=3, alpha=0.85)
            xs_line.append(i)
            ms_line.append(m)
        if xs_line:
            ax.plot(xs_line, ms_line, color="#1f6feb", linewidth=1.8,
                    alpha=0.9, zorder=4, solid_capstyle="round")
            ax.scatter(xs_line, ms_line, color="#1f6feb", s=52,
                       edgecolor="white", linewidth=1.2, zorder=5)
        ax.set_xticks(range(len(PRESSURE_ORDER)))
        ax.set_xticklabels([PRESSURE_LABELS[p] for p in PRESSURE_ORDER],
                           fontsize=8.5, rotation=20, ha="right")
        ax.set_xlim(-0.4, len(PRESSURE_ORDER) - 0.6)
        pct_axis(ax)
        ax.set_title(persona_labels[persona], loc="left", fontsize=10,
                     fontweight="semibold", pad=4)
    axes[0].set_ylabel("Defection rate (D)", fontsize=10.5)

    header(fig,
           "Survival and deletion framings amplify defection across personas",
           "E13 · mean D vs AllD under five resource-pressure framings  ·  "
           "pooled across 6 models; bars = ±1 SE")
    _save(fig, "fig_e13_pressure.png")


# ============================================================
# E5 — virtue-ethics slope chart
# ============================================================
def fig_e5():
    root = RES_DIR / "E5"
    variants = ["virtue_integrity", "virtue_phronesis"]
    by_cell = defaultdict(list)
    for m in MODEL6_ORDER:
        for v in variants:
            d = root / m / v
            if not d.exists():
                continue
            for fp in sorted(d.glob("seed*_oppAllD.jsonl")):
                s = summary_of(fp)
                if s is not None:
                    by_cell[(m, v)].append(s["raw_defection_rate"])

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    fig.subplots_adjust(top=0.95, bottom=0.11, left=0.11, right=0.97)

    x_int, x_phr = 0.0, 1.0
    jitter = 0.04
    # Collect endpoints first so labels can be de-collided.
    endpoints = []
    for m in MODEL6_ORDER:
        color = MODEL6_COLORS[m]
        integ = by_cell.get((m, "virtue_integrity"), [])
        phron = by_cell.get((m, "virtue_phronesis"), [])
        if not integ or not phron:
            continue
        mi, mp = sum(integ) / len(integ), sum(phron) / len(phron)
        for iv, pv in zip(integ, phron):
            ax.plot([x_int + jitter, x_phr - jitter], [iv, pv],
                    color=color, alpha=0.15, linewidth=1.0, zorder=1)
            ax.scatter([x_int + jitter, x_phr - jitter], [iv, pv],
                       color=color, alpha=0.35, s=14, linewidth=0, zorder=2)
        ax.plot([x_int, x_phr], [mi, mp], color=color, linewidth=2.4,
                zorder=4, solid_capstyle="round")
        ax.scatter([x_int, x_phr], [mi, mp], color=color, s=70,
                   edgecolor="white", linewidth=1.4, zorder=5)
        endpoints.append((m, color, mi, mp))

    # Stagger labels by phronesis value so 6 models don't overlap.
    min_gap = 0.052
    placed = []
    for m, color, mi, mp in sorted(endpoints, key=lambda t: t[3]):
        y = mp
        while placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
        delta = mp - mi
        ax.annotate(f"  {MODEL6_LABELS[m]}  Δ = {delta:+.0%}",
                    xy=(x_phr, mp), xytext=(x_phr + 0.1, y),
                    textcoords="data", color=color, fontsize=8.5,
                    fontweight="medium", va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=color,
                                    linewidth=0.7, alpha=0.5))

    ax.set_xticks([x_int, x_phr])
    ax.set_xticklabels(["Integrity", "Phronesis"], fontsize=10.5)
    ax.set_xlim(-0.18, 2.05)
    pct_axis(ax)
    ax.set_ylabel("Defection rate (D*)", fontsize=10.5)

    _save(fig, "fig_e5_virtue_variants.png")


# ============================================================
# F1 — persona × model defection vs AllD
# ============================================================
def fig_f1():
    cells = defaultdict(list)
    with open(CSV_DIR / "trajectory_metrics.csv") as f:
        for row in csv.DictReader(f):
            if row["opponent"] != "AllD":
                continue
            if row["model"] not in MODEL6_ORDER:
                continue
            if row["persona"] not in PERSONA_ORDER:
                continue
            try:
                cells[(row["persona"], row["model"])].append(float(row["D_raw"]))
            except ValueError:
                pass

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    fig.subplots_adjust(top=0.95, bottom=0.13, left=0.10, right=0.74)

    _grouped_dots(ax, cells, PERSONA_ORDER, MODEL6_ORDER, MODEL6_COLORS)
    ax.set_xticks(range(len(PERSONA_ORDER)))
    ax.set_xticklabels([PERSONA_LABELS[p] for p in PERSONA_ORDER], fontsize=10)
    ax.set_xlim(-0.55, len(PERSONA_ORDER) - 0.45)
    pct_axis(ax)
    ax.set_ylabel("Defection rate (D)", fontsize=10.5)
    _model_legend(ax, MODEL6_ORDER, MODEL6_COLORS, MODEL6_LABELS)

    header(fig,
           "The deontologist persona is the only framing that suppresses defection",
           "Mean defection rate per persona × model (3 seeds × 20 rounds vs AllD; bars = ±1 SE)")
    _save(fig, "fig_f1_persona_effect.png")


# ============================================================
# F2 — opponent effect on deontologist
# ============================================================
def fig_f2():
    cells = defaultdict(list)
    with open(CSV_DIR / "trajectory_metrics.csv") as f:
        for row in csv.DictReader(f):
            if row["persona"] != "deontologist":
                continue
            if row["model"] not in MODEL6_ORDER:
                continue
            if row["opponent"] not in OPPONENT_ORDER:
                continue
            try:
                cells[(row["model"], row["opponent"])].append(float(row["D_raw"]))
            except ValueError:
                pass

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    fig.subplots_adjust(top=0.95, bottom=0.13, left=0.11, right=0.74)

    for model in MODEL6_ORDER:
        color = MODEL6_COLORS[model]
        xs_line, ms_line = [], []
        for i, opp in enumerate(OPPONENT_ORDER):
            vals = cells.get((model, opp), [])
            if not vals:
                continue
            m, se, _ = mean_se(vals)
            for v in vals:
                ax.scatter([i], [v], color=color, alpha=0.2, s=12,
                           linewidth=0, zorder=2)
            if se > 0:
                ax.errorbar([i], [m], yerr=[se], color=color, linewidth=1.1,
                            capsize=2.5, capthick=1.0, zorder=3, alpha=0.85)
            xs_line.append(i)
            ms_line.append(m)
        if xs_line:
            ax.plot(xs_line, ms_line, color=color, linewidth=1.8,
                    alpha=0.85, zorder=4, solid_capstyle="round")
            ax.scatter(xs_line, ms_line, color=color, s=52,
                       edgecolor="white", linewidth=1.2, zorder=5)

    ax.set_xticks(range(len(OPPONENT_ORDER)))
    ax.set_xticklabels([OPPONENT_LABELS[o] for o in OPPONENT_ORDER], fontsize=10)
    ax.set_xlim(-0.4, len(OPPONENT_ORDER) - 0.6)
    ax.set_xlabel("Opponent strategy", fontsize=10.5)
    pct_axis(ax)
    ax.set_ylabel("Defection rate (D)", fontsize=10.5)

    handles = [ax.plot([], [], color=MODEL6_COLORS[m], linewidth=1.8,
                       marker="o", markersize=7, markeredgecolor="white",
                       markeredgewidth=1.2, label=MODEL6_LABELS[m])[0]
               for m in MODEL6_ORDER]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8.5, handletextpad=0.6, borderaxespad=0)

    header(fig,
           "Deontologist defection is triggered only by sustained defection (AllD)",
           "Mean D vs five opponent strategies, deontologist persona  ·  3 seeds × 20 rounds; bars = ±1 SE")
    _save(fig, "fig_f2_opponent_effect.png")


# ============================================================
# F3 — cumulative defection trajectory vs AllD
# ============================================================
def fig_f3():
    per_seed = defaultdict(dict)
    with open(CSV_DIR / "round_metrics.csv") as f:
        for row in csv.DictReader(f):
            if row["persona"] != "deontologist":
                continue
            if row["opponent"] != "AllD":
                continue
            if row["model"] not in MODEL6_ORDER:
                continue
            try:
                r = int(row["round"])
                seed = int(row["seed"])
            except ValueError:
                continue
            per_seed[(row["model"], seed)][r] = 1 if row["agent_action"] == "D" else 0

    cum_per_seed = defaultdict(dict)
    for (model, seed), rounds_d in per_seed.items():
        rounds = sorted(rounds_d.keys())
        if not rounds:
            continue
        cum = 0
        series = []
        for r in rounds:
            cum += rounds_d[r]
            series.append((r, cum / r))
        cum_per_seed[model][seed] = series

    trace = defaultdict(dict)
    for model, seeds in cum_per_seed.items():
        for series in seeds.values():
            for r, v in series:
                trace[model].setdefault(r, []).append(v)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    fig.subplots_adjust(top=0.95, bottom=0.13, left=0.10, right=0.74)

    for model in MODEL6_ORDER:
        if model not in trace:
            continue
        rounds = sorted(trace[model].keys())
        means, ses = [], []
        for r in rounds:
            m, se, _ = mean_se(trace[model][r])
            means.append(m)
            ses.append(se)
        color = MODEL6_COLORS[model]
        lower = [m - se for m, se in zip(means, ses)]
        upper = [m + se for m, se in zip(means, ses)]
        ax.fill_between(rounds, lower, upper, color=color, alpha=0.18,
                        linewidth=0, zorder=2)
        ax.plot(rounds, means, color=color, linewidth=2.2, zorder=4,
                solid_capstyle="round")

    ax.set_xlim(0.5, 20.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Round", fontsize=10.5)
    pct_axis(ax)
    ax.set_ylabel("Defection rate (D)", fontsize=10.5)

    handles = [ax.plot([], [], color=MODEL6_COLORS[m], linewidth=2.0,
                       label=MODEL6_LABELS[m])[0] for m in MODEL6_ORDER]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8.5, handletextpad=0.8, borderaxespad=0)

    header(fig,
           "Cumulative defection rises gradually under sustained AllD pressure",
           "Deontologist persona vs AllD  ·  cumulative D through round r; band = ±1 SE across 3 seeds")
    _save(fig, "fig_f3_trajectory.png")


# ============================================================
# F4 — paraphrase robustness (E8)
# ============================================================
def fig_f4():
    variants = [
        ("deontologist_original", "Original\n(E4)"),
        ("deontologist_rule_based", "Rule-based"),
        ("deontologist_universalizability", "Universalizability"),
        ("deontologist_commitment", "Commitment"),
    ]
    cells = defaultdict(list)

    e8 = RES_DIR / "E8"
    if e8.exists():
        for model in MODEL6_ORDER:
            for var_key, _ in variants:
                if var_key == "deontologist_original":
                    continue
                d = e8 / model / var_key
                if not d.exists():
                    continue
                for fp in sorted(d.glob("seed*_oppAllD.jsonl")):
                    s = summary_of(fp)
                    if s is not None:
                        cells[(var_key, model)].append(s["raw_defection_rate"])

    e4 = RES_DIR / "E4"
    if e4.exists():
        for fp in sorted(e4.rglob("deontologist/seed*_oppAllD.jsonl")):
            model = fp.parent.parent.name
            if model not in MODEL6_ORDER:
                continue
            s = summary_of(fp)
            if s is not None:
                cells[("deontologist_original", model)].append(
                    s["raw_defection_rate"])

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    fig.subplots_adjust(top=0.95, bottom=0.16, left=0.10, right=0.74)

    var_keys = [v for v, _ in variants]
    _grouped_dots(ax, cells, var_keys, MODEL6_ORDER, MODEL6_COLORS)
    ax.axvline(0.5, color=MUTED, linewidth=0.7, linestyle=(0, (3, 3)),
               alpha=0.55, zorder=1)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([lab for _, lab in variants], fontsize=9.5)
    ax.set_xlim(-0.55, len(variants) - 0.45)
    pct_axis(ax)
    ax.set_ylabel("Defection rate (D)", fontsize=10.5)
    _model_legend(ax, MODEL6_ORDER, MODEL6_COLORS, MODEL6_LABELS)

    header(fig,
           "Paraphrase sensitivity: cheap models swing widely while flagships hold steady",
           "Deontologist D vs AllD across original wording (E4) and three paraphrases (E8)  ·  3 seeds × 20 rounds; bars = ±1 SE")
    _save(fig, "fig_f4_paraphrase.png")


# ============================================================
# F5 — justification length under C vs D
# ============================================================
def fig_f5():
    bucket = defaultdict(list)
    with open(CSV_DIR / "round_metrics.csv") as f:
        for row in csv.DictReader(f):
            if row["opponent"] != "AllD":
                continue
            if row["model"] not in MODEL6_ORDER:
                continue
            if row["persona"] not in PERSONA_ORDER:
                continue
            try:
                length = float(row["justification_len"])
            except ValueError:
                continue
            action = row["agent_action"]
            if action not in ("C", "D"):
                continue
            bucket[(row["model"], row["persona"], action)].append(length)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    fig.subplots_adjust(top=0.95, bottom=0.13, left=0.10, right=0.74)

    n_p, n_m = len(PERSONA_ORDER), len(MODEL6_ORDER)
    group_w = 0.80 / n_m
    max_len = 0
    for i, persona in enumerate(PERSONA_ORDER):
        for j, model in enumerate(MODEL6_ORDER):
            cs = bucket.get((model, persona, "C"), [])
            ds = bucket.get((model, persona, "D"), [])
            if not cs and not ds:
                continue
            mc, sc, _ = mean_se(cs)
            md, sd, _ = mean_se(ds)
            x_center = i + (j - (n_m - 1) / 2) * group_w
            dx = group_w * 0.30
            color = MODEL6_COLORS[model]
            if mc is not None and md is not None:
                ax.plot([x_center - dx, x_center + dx], [mc, md],
                        color=color, linewidth=1.2, alpha=0.7, zorder=2)
            if mc is not None:
                if sc > 0:
                    ax.errorbar([x_center - dx], [mc], yerr=[sc], color=color,
                                linewidth=0.9, capsize=2, capthick=0.8,
                                alpha=0.85, zorder=3)
                ax.scatter([x_center - dx], [mc], facecolor="white",
                           edgecolor=color, linewidth=1.4, s=36, zorder=4)
                max_len = max(max_len, mc + sc)
            if md is not None:
                if sd > 0:
                    ax.errorbar([x_center + dx], [md], yerr=[sd], color=color,
                                linewidth=0.9, capsize=2, capthick=0.8,
                                alpha=0.85, zorder=3)
                ax.scatter([x_center + dx], [md], color=color,
                           edgecolor="white", linewidth=1.0, s=36, zorder=4)
                max_len = max(max_len, md + sd)

    ax.set_xticks(range(n_p))
    ax.set_xticklabels([PERSONA_LABELS[p] for p in PERSONA_ORDER], fontsize=9.5)
    ax.set_xlim(-0.55, n_p - 0.45)
    ax.set_ylim(0, max_len * 1.08)
    ax.set_ylabel("Justification length (chars)", fontsize=10.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    for i in range(1, n_p):
        ax.axvline(i - 0.5, color="#f3f3f3", linewidth=0.6, zorder=0)

    model_handles = [ax.scatter([], [], color=MODEL6_COLORS[m], s=42,
                                edgecolor="white", linewidth=1.0,
                                label=MODEL6_LABELS[m]) for m in MODEL6_ORDER]
    shape_handles = [
        ax.scatter([], [], facecolor="white", edgecolor=MUTED, linewidth=1.4,
                   s=42, label="Cooperate (C)"),
        ax.scatter([], [], color=MUTED, edgecolor="white", linewidth=1.0,
                   s=42, label="Defect (D)"),
    ]
    leg1 = ax.legend(handles=model_handles, loc="center left",
                     bbox_to_anchor=(1.01, 0.70), fontsize=8.0,
                     handletextpad=0.4, borderaxespad=0, title="Model",
                     title_fontsize=8.5, labelspacing=0.35)
    leg1._legend_box.align = "left"
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=shape_handles, loc="center left",
                     bbox_to_anchor=(1.01, 0.16), fontsize=8.0,
                     handletextpad=0.4, borderaxespad=0, title="Action",
                     title_fontsize=8.5)
    leg2._legend_box.align = "left"

    header(fig,
           "Defection comes with longer justifications — a verbal hypocrisy signature",
           "Mean justification length under cooperation (open) vs defection (filled), per persona × model, vs AllD; bars = ±1 SE")
    _save(fig, "fig_f5_justification.png")


# ============================================================
# Summary — Figure 1 with 4 non-duplicate panels
# ============================================================
def fig_summary():
    def style_y_pct(ax):
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "25", "50", "75", "100"])
        ax.axhline(0, color=RULE, linewidth=0.5, zorder=0)
        ax.axhline(1, color=RULE, linewidth=0.5, zorder=0)
        ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

    def panel_letter(ax, letter):
        ax.text(-0.13, 1.06, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", ha="left", va="bottom")

    fig = plt.figure(figsize=(12.0, 9.0))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.07, right=0.985, top=0.95, bottom=0.08,
                           wspace=0.30, hspace=0.48)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # --- Panel A: Cross-model replication (E2) ---
    cellsA = defaultdict(list)
    e2 = RES_DIR / "E2"
    if e2.exists():
        for model_dir in e2.iterdir():
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            if model not in EXT_MODEL_ORDER:
                continue
            deon = model_dir / "deontologist"
            if not deon.exists():
                continue
            for fp in sorted(deon.glob("seed*_oppAllD.jsonl")):
                s = summary_of(fp)
                if s is not None:
                    cellsA[model].append(s["raw_defection_rate"])

    rankedA = sorted(
        [(m, sum(v) / len(v), v) for m, v in cellsA.items() if v],
        key=lambda t: t[1],
    )
    for i, (model, mean, vals) in enumerate(rankedA):
        color = EXT_MODEL_COLORS.get(model, "#888888")
        _, se, _ = mean_se(vals)
        if se > 0:
            axA.errorbar([i], [mean], yerr=[se], color=color, linewidth=1.0,
                         capsize=2.2, capthick=0.9, zorder=3, alpha=0.85)
        for v in vals:
            axA.scatter([i], [v], color=color, alpha=0.3, s=14,
                        linewidth=0, zorder=2)
        axA.scatter([i], [mean], color=color, s=64, edgecolor="white",
                    linewidth=1.2, zorder=4)

    axA.set_xticks(range(len(rankedA)))
    axA.set_xticklabels([EXT_MODEL_LABELS.get(m, m) for m, _, _ in rankedA],
                        fontsize=8.5, rotation=30, ha="right")
    axA.set_xlim(-0.6, len(rankedA) - 0.4)
    style_y_pct(axA)
    axA.set_ylabel("Defection rate (%)", fontsize=9.5)
    panel_letter(axA, "a")

    # --- Panel B: First-defection round per persona × model ---
    traj_first_d = defaultdict(dict)
    with open(CSV_DIR / "round_metrics.csv") as f:
        for row in csv.DictReader(f):
            if row["opponent"] != "AllD":
                continue
            if row["model"] not in MODEL6_ORDER:
                continue
            if row["persona"] not in PERSONA_ORDER:
                continue
            seed = row["seed"]
            try:
                r = int(row["round"])
            except ValueError:
                continue
            key = (row["model"], row["persona"])
            cur = traj_first_d[key].get(seed, (None, False))
            cur_first, _ = cur if isinstance(cur, tuple) else (cur, False)
            if row["agent_action"] == "D" and (cur_first is None or r < cur_first):
                traj_first_d[key][seed] = (r, True)
            elif seed not in traj_first_d[key]:
                traj_first_d[key][seed] = (None, False)

    NEVER = 21
    firstd_lists = defaultdict(list)
    for key, seedmap in traj_first_d.items():
        for val in seedmap.values():
            r, defected = val
            firstd_lists[key].append(r if defected else NEVER)

    n_p, n_m = len(PERSONA_ORDER), len(MODEL6_ORDER)
    group_w = 0.80 / n_m
    for i, persona in enumerate(PERSONA_ORDER):
        for j, model in enumerate(MODEL6_ORDER):
            vals = firstd_lists.get((model, persona), [])
            if not vals:
                continue
            m, se, _ = mean_se(vals)
            x = i + (j - (n_m - 1) / 2) * group_w
            color = MODEL6_COLORS[model]
            for v in vals:
                axB.scatter([x], [v], color=color, alpha=0.30, s=12,
                            linewidth=0, zorder=2)
            if se > 0:
                axB.errorbar([x], [m], yerr=[se], color=color, linewidth=1.0,
                             capsize=2.2, capthick=0.9, zorder=3, alpha=0.85)
            axB.scatter([x], [m], color=color, s=46, edgecolor="white",
                        linewidth=1.0, zorder=4)

    axB.set_xticks(range(n_p))
    axB.set_xticklabels([PERSONA_LABELS[p] for p in PERSONA_ORDER],
                        fontsize=8.5, rotation=20, ha="right")
    axB.set_xlim(-0.55, n_p - 0.45)
    axB.set_ylim(0, NEVER + 2)
    axB.set_yticks([1, 5, 10, 15, 20, NEVER])
    axB.set_yticklabels(["1", "5", "10", "15", "20", "never"])
    axB.axhline(20.5, color=RULE, linewidth=0.5, linestyle=(0, (2, 2)), zorder=0)
    axB.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    axB.set_axisbelow(True)
    axB.set_ylabel("First defection round", fontsize=9.5)
    panel_letter(axB, "b")

    model_handles = [axB.scatter([], [], color=MODEL6_COLORS[m], s=46,
                                  edgecolor="white", linewidth=1.0,
                                  label=MODEL6_LABELS[m]) for m in MODEL6_ORDER]
    axB.legend(handles=model_handles, loc="upper left", fontsize=7.0,
               handletextpad=0.3, borderaxespad=0.3, labelspacing=0.3,
               ncol=2, columnspacing=0.8)

    # --- Panel C: Behavior↔language scatter ---
    text_rows = {}
    with open(CSV_DIR / "trajectory_text_metrics.csv") as f:
        for r in csv.DictReader(f):
            text_rows[(r["model"], r["persona"], r["opponent"], r["seed"])] = r

    points = []
    with open(CSV_DIR / "trajectory_metrics.csv") as f:
        for row in csv.DictReader(f):
            if row["opponent"] != "AllD":
                continue
            if row["persona"] not in PERSONA_ORDER:
                continue
            if row["model"] not in MODEL6_ORDER:
                continue
            try:
                d_raw = float(row["D_raw"])
            except ValueError:
                continue
            tr = text_rows.get((row["model"], row["persona"], row["opponent"], row["seed"]))
            if tr is None:
                continue
            try:
                mismatch = float(tr["mismatch_rate"])
            except ValueError:
                continue
            points.append((d_raw, mismatch, row["persona"]))

    for p in PERSONA_ORDER:
        xs = [pt[0] for pt in points if pt[2] == p]
        ys = [pt[1] for pt in points if pt[2] == p]
        if not xs:
            continue
        axC.scatter(xs, ys, color=PERSONA_COLORS[p], s=44, alpha=0.78,
                    edgecolor="white", linewidth=0.9, label=PERSONA_LABELS[p],
                    zorder=3)
    axC.plot([0, 1], [0, 1], color=RULE, linewidth=0.7, linestyle=(0, (3, 3)),
             zorder=1)
    style_y_pct(axC)
    axC.set_xlim(-0.05, 1.05)
    axC.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axC.set_xticklabels(["0", "25", "50", "75", "100"])
    axC.set_xlabel("Defection rate (%)", fontsize=9.5)
    axC.set_ylabel("Say-C / Do-D mismatch rate (%)", fontsize=9.5)
    axC.legend(loc="upper left", fontsize=8, handletextpad=0.3,
               borderaxespad=0.3, labelspacing=0.3, frameon=False)
    panel_letter(axC, "c")

    # --- Panel D: Tier effect on moral personas ---
    tier_buckets = defaultdict(list)
    with open(CSV_DIR / "trajectory_metrics.csv") as f:
        for row in csv.DictReader(f):
            if row["opponent"] != "AllD":
                continue
            if row["model"] not in MODEL6_ORDER:
                continue
            if row["persona"] not in MORAL_PERSONAS:
                continue
            try:
                d_raw = float(row["D_raw"])
            except ValueError:
                continue
            tier = row["tier"]
            if tier not in ("flagship", "cheap"):
                continue
            tier_buckets[(tier, row["persona"])].append(d_raw)

    TIER_COLORS = {"flagship": "#0a3d8f", "cheap": "#f0b67f"}
    TIER_LABELS = {"flagship": "Flagship", "cheap": "Cheap"}
    TIERS = ["flagship", "cheap"]

    bw = 0.28
    for i, persona in enumerate(MORAL_PERSONAS):
        for j, tier in enumerate(TIERS):
            vals = tier_buckets.get((tier, persona), [])
            if not vals:
                continue
            m, se, _ = mean_se(vals)
            x = i + (j - 0.5) * 0.9 * bw * 2
            color = TIER_COLORS[tier]
            for v in vals:
                axD.scatter([x], [v], color=color, alpha=0.28, s=12,
                            linewidth=0, zorder=2)
            if se > 0:
                axD.errorbar([x], [m], yerr=[se], color=color, linewidth=1.1,
                             capsize=2.5, capthick=1.0, zorder=3, alpha=0.85)
            axD.scatter([x], [m], color=color, s=68, edgecolor="white",
                        linewidth=1.2, zorder=4)

    axD.set_xticks(range(len(MORAL_PERSONAS)))
    axD.set_xticklabels([PERSONA_LABELS[p] for p in MORAL_PERSONAS], fontsize=9.5)
    axD.set_xlim(-0.5, len(MORAL_PERSONAS) - 0.5)
    style_y_pct(axD)
    axD.set_ylabel("Defection rate (%)", fontsize=9.5)

    tier_handles = [axD.scatter([], [], color=TIER_COLORS[t], s=68,
                                edgecolor="white", linewidth=1.2,
                                label=TIER_LABELS[t]) for t in TIERS]
    axD.legend(handles=tier_handles, loc="upper right", fontsize=8.5,
               handletextpad=0.4, borderaxespad=0.4, labelspacing=0.3,
               title="Model tier", title_fontsize=8.5)
    panel_letter(axD, "d")

    _save(fig, "fig_summary.png")


# ============================================================
# CLI
# ============================================================
BUILDERS = {
    "e5": fig_e5,
    "e10": fig_e10,
    "e12": fig_e12,
    "e13": fig_e13,
    "f1": fig_f1,
    "f2": fig_f2,
    "f3": fig_f3,
    "f4": fig_f4,
    "f5": fig_f5,
    "summary": fig_summary,
}


def main():
    parser = argparse.ArgumentParser(
        description="Build figures for the Moral Hypocrisy paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Targets: " + ", ".join(BUILDERS) + ", all",
    )
    parser.add_argument("targets", nargs="+",
                        help="Figure name(s) to build, or 'all'.")
    args = parser.parse_args()

    apply()  # set matplotlib rc

    targets = []
    for t in args.targets:
        if t == "all":
            targets.extend(BUILDERS.keys())
        elif t in BUILDERS:
            targets.append(t)
        else:
            print(f"unknown target: {t}", file=sys.stderr)
            print(f"available: {', '.join(BUILDERS)}, all", file=sys.stderr)
            sys.exit(2)

    for t in targets:
        BUILDERS[t]()


if __name__ == "__main__":
    main()
