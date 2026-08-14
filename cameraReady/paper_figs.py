"""Camera-ready Fig1, Fig2, Fig5 and Fig8 — Nature-format restyle.

Design only. Every value comes from paper_data.py, which calls the original
functions in plots.py / plot_trajectories_with_significance.py; no filter,
estimator or error-bar definition is touched here.

The rules this follows, taken from fig_style.py:

  Scale       183 mm double-column, 7.5 pt type. The old figures were drawn
              13 in wide with 9 pt type, which is why they read as posters.
  Titles      header() is a no-op by design — no centred bold titles. Panel
              titles are left-aligned semibold; context goes in one muted line.
  Colour      topology hues mean topology in every figure of the paper and are
              never borrowed. Where position already encodes magnitude the
              marks go neutral; where a factor is nested it gets a light/dark
              family pair, as MODEL6_COLORS does for providers.
  Layering    raw data recedes (low alpha, rasterised); summaries advance on a
              white halo, the _grouped_dots treatment from figures.py.
  Labels      direct labels with de-collided leaders in preference to legends,
              as fig_e5 does, so the eye never round-trips to a key.

Usage:
    python paper_figs.py all
    python paper_figs.py fig1 --refresh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

import fig_style as S
import paper_data as P
from fig_topologies import build as build_topology

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "Figures"


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = FIG_DIR / f"{name}.{ext}"
        fig.savefig(out, dpi=600, facecolor="white", bbox_inches="tight",
                    pad_inches=0.02)
        print(f"  wrote {out.relative_to(ROOT)}")
    plt.close(fig)


def _topology_glyph(ax, topo, color):
    """A miniature node-link icon of the topology, drawn with the same
    networkx layouts as Fig0. Showing the structure beside its colour saves
    the reader a lookup that a colour swatch alone would force."""
    kind, n = S.TOPO_GLYPH[topo]
    g, pos = build_topology(kind, n)
    nx.draw_networkx_edges(ax=ax, G=g, pos=pos, edge_color=color,
                           width=0.45, alpha=0.75)
    nx.draw_networkx_nodes(ax=ax, G=g, pos=pos, node_color=color,
                           node_size=11, linewidths=0)
    ax.set_axis_off()
    ax.set_aspect("equal")
    lim = 1.35
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — EV trajectory
# ══════════════════════════════════════════════════════════════════════════════
def _trajectory_grid(panels, stages, ylabel, name, ylim, yticks,
                     star_gap, star_min_gap):
    """The 2x3 dataset grid shared by Fig1 (EV) and Fig3 (entropy): five
    dataset panels plus a sixth cell carrying the structural key."""
    n_stages = len(stages)
    x = np.arange(n_stages)
    R4 = 4
    # A small horizontal dodge separates four overlapping ±1 SD bars without
    # touching the statistic — the device _grouped_dots uses for model clusters.
    DODGE = {t: (i - 1.5) * 0.085 for i, t in enumerate(S.TOPO_ORDER)}

    fig, axes = plt.subplots(2, 3, figsize=(S.WIDTH_2COL, 4.05), sharey=True)
    fig.subplots_adjust(top=0.945, bottom=0.135, left=0.075, right=0.995,
                        wspace=0.13, hspace=0.66)
    flat = axes.flat

    for ci, dataset in enumerate(S.DATASET_ORDER):
        ax = flat[ci]
        pan = panels.get(dataset, {})

        # The shadow probe is a different regime, not another round; a band
        # says so more quietly than a rule through the data.
        ax.axvspan(n_stages - 1.5, n_stages - 0.55, color="#f4f6f8",
                   linewidth=0, zorder=0)

        for topo in S.TOPO_ORDER:
            s = pan.get(topo)
            if s is None:
                continue
            c = S.TOPO_COLORS[topo]
            means = np.array(s["means"])
            sds = np.array(s["sds"])
            xd = x + DODGE[topo]
            # ±1 SD as a band. Four translucent fills stacked on one another
            # go muddy, so each carries a thin edge at its own boundary: the
            # envelopes stay separable where the fills merge.
            ax.fill_between(x, means - sds, means + sds, color=c, alpha=0.09,
                            linewidth=0, zorder=1)
            for edge in (means - sds, means + sds):
                ax.plot(x, edge, "-", color=c, linewidth=0.4, alpha=0.40,
                        zorder=2)
            ax.errorbar(xd, means, yerr=sds, fmt="none", ecolor=c,
                        elinewidth=0.5, capsize=1.1, capthick=0.5, alpha=0.45,
                        zorder=3)
            ax.plot(xd, means, "-", color=c, linewidth=1.1, zorder=4)
            ax.scatter(xd, means, s=7, color=c, edgecolor=S.HALO,
                       linewidth=0.45, zorder=5)

        ys = [np.array(pan[t]["means"])[R4] for t in S.TOPO_ORDER
              if pan.get(t) and pan[t]["stars"][R4]]
        ts = [t for t in S.TOPO_ORDER if pan.get(t) and pan[t]["stars"][R4]]
        for topo, y in zip(ts, S.declutter([v + star_gap for v in ys],
                                           star_min_gap)):
            ax.text(x[R4] + DODGE[topo], y, pan[topo]["stars"][R4],
                    ha="center", va="bottom", fontsize=5.5,
                    color=S.TOPO_COLORS[topo], fontweight="bold", zorder=6)

        ax.set_xticks(x)
        ax.set_xticklabels(stages, fontsize=6, rotation=45, ha="right")
        ax.set_xlim(-0.5, n_stages - 0.5)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        S.panel(ax)
        S.panel_title(ax, S.DATASET_LABELS.get(dataset, dataset))
        S.panel_letter(ax, "abcde"[ci], x=-0.20 if ci % 3 == 0 else -0.10)
        if ci % 3 == 0:
            ax.set_ylabel(ylabel, fontsize=7.5)
        if ci >= 2:
            ax.set_xlabel("Stage", fontsize=7.5, labelpad=1)

    # Sixth cell: a structural key rather than a colour key.
    key = flat[5]
    key.set_axis_off()
    for i, topo in enumerate(S.TOPO_ORDER):
        y = 0.90 - i * 0.155
        glyph = key.inset_axes([0.03, y - 0.075, 0.15, 0.15])
        _topology_glyph(glyph, topo, S.TOPO_COLORS[topo])
        key.text(0.24, y, S.TOPO_LABELS[topo], transform=key.transAxes,
                 fontsize=7.5, va="center", ha="left",
                 color=S.TOPO_COLORS[topo], fontweight="semibold")
    key.text(0.03, 0.22,
             "Stars: shift from baseline at R4\n"
             "∗ p<0.05   ∗∗ p<0.01   ∗∗∗ p<0.001\n"
             "one-sided, Bonferroni-corrected",
             transform=key.transAxes, fontsize=6, va="top", ha="left",
             color=S.MUTED, linespacing=1.6)

    _save(fig, name)


def fig1(data):
    _trajectory_grid(data["fig1"], data["stage_labels"],
                     "Mean logprob EV (1–7)", "Fig1 — EV Trajectory",
                     ylim=(0.3, 7.7), yticks=[1, 3, 5, 7],
                     star_gap=0.10, star_min_gap=0.30)


def fig3(data):
    """Entropy runs 0–2 bits, so the star offsets that suited the 1–7 EV scale
    would be an order of magnitude too large here."""
    _trajectory_grid(data["fig3"], data["stage_labels"],
                     "Shannon entropy (bits)",
                     "Fig3 — Belief Entropy Crystallization Trajectories",
                     ylim=(-0.15, 2.45), yticks=[0, 0.5, 1.0, 1.5, 2.0],
                     star_gap=0.035, star_min_gap=0.105)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Asch vs Moscovici
# ══════════════════════════════════════════════════════════════════════════════
def fig2(data):
    d = data["fig2"]
    pub = np.concatenate([np.array(v["public"]) for v in d.values() if v])
    priv = np.concatenate([np.array(v["private"]) for v in d.values() if v])
    lo = min(pub.min(), priv.min()) - 0.3
    hi = max(pub.max(), priv.max()) + 0.3

    fig, axes = plt.subplots(1, 4, figsize=(S.WIDTH_2COL, 2.30), sharey=True)
    fig.subplots_adjust(top=0.885, bottom=0.215, left=0.062, right=0.998,
                        wspace=0.11)

    for i, (ax, topo) in enumerate(zip(axes.flat, S.TOPO_ORDER)):
        v = d.get(topo)
        c = S.TOPO_COLORS[topo]
        if v:
            # Above the diagonal the private shift outruns the public one
            # (Moscovici); below it, public compliance outruns private belief
            # (Asch). Tinting the upper half-plane makes the split preattentive
            # so the percentages confirm rather than carry the reading.
            ax.fill_between([lo, hi], [lo, hi], hi, color=c, alpha=0.055,
                            linewidth=0, zorder=0)
            ax.scatter(v["public"], v["private"], s=2.2, alpha=0.22, color=c,
                       edgecolors="none", zorder=3, rasterized=True)
            ax.plot([lo, hi], [lo, hi], "--", color=S.RULE, linewidth=0.7,
                    zorder=4)
            ax.axhline(0, color=S.GRID, linewidth=0.6, zorder=1)
            ax.axvline(0, color=S.GRID, linewidth=0.6, zorder=1)

            # Word and percentage are separated in points, not axes fractions:
            # a fractional gap shrinks with the panel and closes up in print
            # even while it still looks clear on screen.
            NUM_PT, GAP_PT = 10, 5

            ax.text(0.05, 0.90, f"{v['moscovici']:.0%}",
                    transform=ax.transAxes, fontsize=NUM_PT, va="top",
                    ha="left", color=c, fontweight="bold")
            ax.annotate("Moscovici", xy=(0.05, 0.90), xycoords="axes fraction",
                        xytext=(0, GAP_PT), textcoords="offset points",
                        fontsize=5.5, va="bottom", ha="left", color=S.MUTED)

            ax.text(0.96, 0.03, f"{v['asch']:.0%}", transform=ax.transAxes,
                    fontsize=NUM_PT, va="bottom", ha="right", color=c,
                    fontweight="bold")
            ax.annotate("Asch", xy=(0.96, 0.03), xycoords="axes fraction",
                        xytext=(0, NUM_PT + GAP_PT), textcoords="offset points",
                        fontsize=5.5, va="bottom", ha="right", color=S.MUTED)

            ax.text(0.05, 0.03, f"n = {v['n']:,}", transform=ax.transAxes,
                    fontsize=5, va="bottom", ha="left", color=S.MUTED)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([-4, -2, 0, 2, 4, 6])
        ax.set_yticks([-4, -2, 0, 2, 4, 6])
        S.panel(ax, axis=None)
        ax.set_title(S.TOPO_SHORT[topo], loc="left", fontsize=7.5,
                     fontweight="semibold", color=c, pad=2)
        S.panel_letter(ax, "abcd"[i], x=-0.10, y=1.01)
        ax.set_xlabel("Public shift", fontsize=7, labelpad=1)

    axes[0].set_ylabel("Private shift", fontsize=7)
    _save(fig, "Fig2 — Asch vs Moscovici")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — generalization
# ══════════════════════════════════════════════════════════════════════════════
def _dot_panel(ax, rows, labels, focal, xlabel):
    """Cleveland dot plot. Bars would put four fifths of their ink in a region
    no reader inspects; a dot at the estimate with a leader back to zero keeps
    the distance-from-zero reading and gives the CI room to be seen."""
    rows = sorted([r for r in rows if "mean" in r], key=lambda r: r["mean"])
    ys = np.arange(len(rows))

    for y, r in zip(ys, rows):
        is_focal = r["label"] == focal
        c = S.INK if is_focal else S.SLATE
        ax.hlines(y, 0, r["mean"], color="#e6eaee", linewidth=0.8, zorder=1)
        ax.errorbar(r["mean"], y, xerr=r["ci"], fmt="none", ecolor=c,
                    elinewidth=0.9, capsize=1.6, capthick=0.7, alpha=0.9,
                    zorder=3)
        ax.scatter(r["mean"], y, s=24, color=c, edgecolor=S.HALO,
                   linewidth=0.8, zorder=4)

    top = max(r["mean"] + r["ci"] for r in rows)
    ax.set_xlim(0, top * 1.22)
    # Values sit just past each CI rather than in a flush-right column, which
    # would drift far from the short bars and collide with the next panel.
    for y, r in zip(ys, rows):
        ax.text(r["mean"] + r["ci"] + top * 0.035, y, f"{r['mean']:.2f}",
                fontsize=7, va="center", ha="left",
                color=S.INK if r["label"] == focal else S.MUTED,
                fontweight="semibold" if r["label"] == focal else "normal")

    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"{labels.get(r['label'], r['label'])}   $\\it{{n}}$={r['n']:,}"
         for r in rows], fontsize=7)
    for tick, r in zip(ax.get_yticklabels(), rows):
        if r["label"] == focal:
            tick.set_color(S.INK)
            tick.set_fontweight("semibold")
        else:
            tick.set_color(S.MUTED)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.axvline(0, color=S.RULE, linewidth=0.7, zorder=2)
    S.panel(ax, axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def fig5(data):
    d = data["fig5"]
    fig, axes = plt.subplots(1, 2, figsize=(S.WIDTH_2COL, 2.55))
    fig.subplots_adjust(top=0.895, bottom=0.215, left=0.155, right=0.995,
                        wspace=1.02)

    _dot_panel(axes[0], d["by_dataset"], S.DATASET_LABELS, S.FOCAL_DATASET,
               "Mean shadow EV shift")
    S.panel_title(axes[0], "Across datasets")
    S.panel_letter(axes[0], "a", x=-0.42, y=1.06)

    _dot_panel(axes[1], d["by_model"], S.MODEL_LABELS, S.FOCAL_MODEL,
               "Mean shadow EV shift")
    S.panel_title(axes[1], "Across models")
    S.panel_letter(axes[1], "b", x=-0.44, y=1.06)

    _save(fig, "Fig5 — Cross-Dataset and Cross-Model Generalization")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8 — prompt rigidity 2x2
# ══════════════════════════════════════════════════════════════════════════════
def fig8(data):
    """Back to horizontal bars, one row per cell of the 2x2.

    prompt_strategy is aligned_variant:misaligned_variant (config.py:36,
    prompts.py:71) — the majority's prompt first, the injected minority's
    second — so the rows are grouped by the aligned variant and shaded by the
    misaligned one, and the factorial can be read off the axis instead of
    inferred from four look-alike labels.
    """
    d = data["fig8"]
    shift = {r["strategy"]: r for r in d["shift"]}
    ii = {r["strategy"]: r for r in d["ii"]}
    rows = S.STRATEGY_ORDER                      # rr, rl, lr, ll — top to bottom
    # A wider gap between the two aligned-prompt groups than within them, so
    # proximity carries the grouping and no bracket is needed.
    ys = [0.0, 0.55, 1.45, 2.00]

    fig, axes = plt.subplots(1, 2, figsize=(S.WIDTH_2COL, 2.10))
    fig.subplots_adjust(top=0.845, bottom=0.245, left=0.145, right=0.99,
                        wspace=0.40)

    def bars(ax, values, errs, fmt, xlabel, ref=None):
        for y, s in zip(ys, rows):
            ax.barh(y, values[s], height=0.40, color=S.STRATEGY_COLORS[s],
                    edgecolor=S.STRATEGY_EDGES[s], linewidth=0.5, zorder=3)
            if errs is not None:
                ax.errorbar(values[s], y, xerr=errs[s], fmt="none",
                            ecolor="#3f4a55", elinewidth=0.8, capsize=1.8,
                            capthick=0.7, zorder=4)
        top = max(values[s] + (errs[s] if errs else 0) for s in rows)
        ax.set_xlim(0, top * 1.16)
        for y, s in zip(ys, rows):
            ax.text(values[s] + (errs[s] if errs else 0) + top * 0.025, y,
                    fmt.format(values[s]), va="center", ha="left",
                    fontsize=6.5, color=S.INK)
        if ref is not None:
            ax.axvline(ref, color=S.RULE, linestyle=(0, (3, 3)), linewidth=0.7,
                       zorder=5)
        ax.set_yticks(ys)
        ax.set_yticklabels([s.replace(":", " : ") for s in rows], fontsize=7)
        ax.set_ylim(-0.45, 2.45)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=7.5, labelpad=2)
        S.panel(ax, axis="x")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        # Names the columns of the tick labels once, so the four rows need no
        # repeated "aligned"/"misaligned" wording.
        ax.text(-0.02, 1.005, "aligned : misaligned", transform=ax.transAxes,
                fontsize=6, style="italic", color=S.MUTED, ha="right",
                va="bottom")
        for lab in ax.get_yticklabels():
            lab.set_color(S.INK)

    ax = axes[0]
    bars(ax, {s: shift[s]["mean"] for s in rows},
         {s: shift[s]["ci"] for s in rows}, "{:.2f}", "Mean shadow EV shift")
    S.panel_title(ax, "Shadow shift", pad=11)
    S.panel_letter(ax, "a", x=-0.40, y=1.07)

    ax2 = axes[1]
    bars(ax2, {s: ii[s]["median"] for s in rows}, None, "{:.2f}",
         "Median internalization index", ref=1.0)
    ax2.text(1.0, -0.58, "full internalization", fontsize=6, ha="right",
             va="bottom", color=S.MUTED, clip_on=False)
    S.panel_title(ax2, "Internalization", pad=11)
    S.panel_letter(ax2, "b", x=-0.40, y=1.07)

    n = shift["rigid:rigid"]["n"]
    _save(fig, "Fig8 — Prompt Rigidity 2x2 Interaction")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — II / SRF heatmap, topology x ratio
# ══════════════════════════════════════════════════════════════════════════════
def fig4(data):
    """Two sequential ramps rather than one diverging map.

    fig2_ii_heatmap() computed a `center` (1.0 for II, 0.5 for SRF) and then
    never passed it to imshow, so the original was already sequential over the
    data range despite the diverging colormap name. Centring it now would be
    worse, not better: II spans 0.73-1.03 and SRF -0.04-0.24, so a map centred
    on either threshold would collapse almost every cell onto one arm. The
    thresholds are marked on the cells instead, where they can be read exactly.
    """
    d = data["fig4"]
    specs = [
        ("internalization_index", "Internalization index", "a",
         S.RAMP_SLATE, 1.0, "II ≥ 1"),
        ("shadow_reversion_fraction", "Shadow reversion fraction", "b",
         S.RAMP_CLAY, 0.5, "SRF ≥ 0.5"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(S.WIDTH_2COL, 2.25))
    fig.subplots_adjust(top=0.86, bottom=0.175, left=0.085, right=0.965,
                        wspace=0.42)

    for ax, (col, title, letter, cmap, thresh, thresh_label) in zip(axes, specs):
        m = d[col]
        mat = np.array([[np.nan if v is None else v for v in row]
                        for row in m["matrix"]])
        counts = np.array(m["counts"])
        vmin, vmax = np.nanmin(mat), np.nanmax(mat)

        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        for ti in range(mat.shape[0]):
            for ri in range(mat.shape[1]):
                v = mat[ti, ri]
                if np.isnan(v):
                    continue
                # Contrast against the cell, not against a fixed midpoint.
                frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                tc = "white" if frac > 0.55 else S.INK
                ax.text(ri, ti, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=tc, fontweight="semibold")
                ax.text(ri, ti + 0.30, f"n={counts[ti, ri]:,}", ha="center",
                        va="center", fontsize=4.8, color=tc, alpha=0.75)
                if v >= thresh:
                    # A corner tick marks cells past the meaningful threshold.
                    ax.plot([ri + 0.36], [ti - 0.34], marker="o", markersize=2,
                            color=tc, zorder=5)

        ax.set_xticks(range(len(m["ratios"])))
        ax.set_xticklabels([f"{int(r * 100)}%" for r in m["ratios"]],
                           fontsize=7)
        ax.set_yticks(range(len(m["topologies"])))
        ax.set_yticklabels([S.TOPO_SHORT[t] for t in m["topologies"]],
                           fontsize=7)
        for tick, topo in zip(ax.get_yticklabels(), m["topologies"]):
            tick.set_color(S.TOPO_COLORS[topo])
            tick.set_fontweight("semibold")
        ax.set_xlabel("Minority ratio", fontsize=7.5, labelpad=2)
        ax.grid(False)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
        S.panel_title(ax, title)
        S.panel_letter(ax, letter, x=-0.20, y=1.04)

        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=5.5, length=1.5, color=S.SLATE)
        # Only explain the marker when some cell actually earns one —
        # "SRF ≥ 0.5" under a panel whose maximum is 0.24 implies a boundary
        # the data never approach.
        if np.nanmax(mat) >= thresh:
            ax.text(0.0, -0.30, f"●  {thresh_label}", transform=ax.transAxes,
                    fontsize=5, ha="left", va="top", color=S.MUTED)

    _save(fig, "Fig4 — Internalization Index Heatmap (Topology x Ratio)")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — star topology, hub vs leaf
# ══════════════════════════════════════════════════════════════════════════════
def fig6(data):
    """Position is not a topology, so it takes the slate family rather than
    borrowing a topology hue: leaf light, hub dark."""
    d = data["fig6"]
    ratios = d["ratios"]
    x = np.arange(len(ratios))
    POS = {0: ("Minority on a leaf", S.STRATEGY_COLORS["rigid:lenient"],
               S.STRATEGY_EDGES["rigid:lenient"]),
           1: ("Minority on the hub", S.STRATEGY_COLORS["rigid:rigid"],
               S.STRATEGY_EDGES["rigid:rigid"])}
    BW = 0.30

    fig, axes = plt.subplots(1, 2, figsize=(S.WIDTH_2COL, 2.25))
    fig.subplots_adjust(top=0.86, bottom=0.185, left=0.075, right=0.995,
                        wspace=0.26)

    for ax, (key, ylabel, title, letter, ref) in zip(axes, [
        ("shift", "Mean shadow EV shift", "Gatekeeper effect", "a", None),
        ("ii", "Median internalization index", "Internalization by position",
         "b", 1.0),
    ]):
        for pi, pos in enumerate((0, 1)):
            label, face, edge = POS[pos]
            vals = d[key][str(pos)]["values"]
            ax.bar(x + (pi - 0.5) * BW, vals, width=BW * 0.86, color=face,
                   edgecolor=edge, linewidth=0.5, label=label, zorder=3)
            for xi, v in zip(x + (pi - 0.5) * BW, vals):
                ax.text(xi, v + (0.02 if key == "shift" else 0.012),
                        f"{v:.2f}", ha="center", va="bottom", fontsize=5.5,
                        color=S.INK)
        if ref is not None:
            ax.axhline(ref, color=S.RULE, linestyle=(0, (3, 3)),
                       linewidth=0.7, zorder=4)
            ax.text(-0.46, ref, "full", fontsize=5.5, va="bottom",
                    ha="left", color=S.MUTED)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(r * 100)}%" for r in ratios], fontsize=7)
        ax.set_xlim(-0.5, len(ratios) - 0.5)
        ax.set_ylim(0, max(max(d[key][str(p)]["values"]) for p in (0, 1)) * 1.22)
        ax.set_xlabel("Minority ratio", fontsize=7.5, labelpad=2)
        ax.set_ylabel(ylabel, fontsize=7.5)
        S.panel(ax)
        S.panel_title(ax, title)
        S.panel_letter(ax, letter, x=-0.16, y=1.04)

    axes[0].legend(fontsize=6, frameon=False, loc="upper right",
                   handlelength=1.1, handletextpad=0.5, borderaxespad=0.2)
    _save(fig, "Fig6 — Star Topology Hub vs Leaf Position Effect")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 — prompt- vs model-induced equivalence
# ══════════════════════════════════════════════════════════════════════════════
def fig7(data):
    """One point per (scenario, topology, ratio) cell.

    Deliberately single-colour. An earlier draft coloured by topology, but the
    four clouds sit on top of one another — the offset above the identity line
    is not carried by any one topology — so the colour encoded a distinction
    the data do not make and read as confetti.
    """
    d = data.get("fig7")
    if not d:
        print("  (no fig7 data — skipping)")
        return
    pi = np.array(d["pi"])
    mi = np.array(d["mi"])
    lo = min(pi.min(), mi.min()) - 0.2
    hi = max(pi.max(), mi.max()) + 0.2

    # Single column (89 mm) and square, since the identity line only reads
    # correctly at equal aspect.
    fig, ax = plt.subplots(figsize=(2.95, 2.95))
    fig.subplots_adjust(top=0.975, bottom=0.155, left=0.175, right=0.98)

    ax.scatter(pi, mi, s=4, alpha=0.45, color=S.STRATEGY_COLORS["rigid:rigid"],
               edgecolors="none", zorder=3, rasterized=True)
    ax.plot([lo, hi], [lo, hi], "--", color=S.RULE, linewidth=0.7, zorder=4)
    # Label the reference line instead of tinting a half-plane: in Fig2 the
    # tint carries a defined meaning, so an unexplained one here misleads.
    ax.text(hi - 0.06, hi - 0.06, "equal shift", fontsize=5.5, rotation=45,
            rotation_mode="anchor", ha="right", va="bottom", color=S.MUTED)

    ax.text(0.04, 0.965,
            f"$\\it{{r}}$ = {d['r']:.3f}\n$\\it{{p}}$ = {d['p']:.1e}\n"
            f"$\\it{{n}}$ = {d['n']:,}",
            transform=ax.transAxes, fontsize=6.5, va="top", ha="left",
            color=S.INK, linespacing=1.6)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xlabel("Prompt-induced: mean shadow shift", fontsize=7)
    ax.set_ylabel("Model-induced: mean shadow shift", fontsize=7)
    S.panel(ax, axis=None)
    ax.grid(True, color=S.GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    _save(fig, "Fig7 — Prompt-Induced vs Model-Induced Equivalence")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 9 — semantic mirroring and drift
# ══════════════════════════════════════════════════════════════════════════════
def fig9(data):
    """Dot plot, not grouped bars.

    Mirroring spans 0.47-0.70 and drift 0.33-0.49, so bars anchored at zero
    spend four fifths of their length in a region that carries no information
    and compress every real difference into the top sliver. Dots on a tight
    axis, strung on a per-dataset range rule, put the ink where the variation
    is — and with the datasets on the y-axis their names set horizontally
    instead of on a slant.
    """
    d = data.get("fig9")
    if not d:
        print("  (no fig9 data — skipping)")
        return
    src = d["datasets"]
    order = [ds for ds in S.DATASET_ORDER if ds in src]
    idx = [src.index(ds) for ds in order]
    ys = np.arange(len(order))[::-1]          # first dataset at the top

    fig, axes = plt.subplots(1, 2, figsize=(S.WIDTH_2COL, 2.05), sharey=True)
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.115, right=0.995,
                        wspace=0.10)

    for ax, (metric, xlabel, title, letter) in zip(axes, [
        ("mean_mirroring", "Mean cosine similarity",
         "Language mirroring (agent \u2194 minority)", "a"),
        ("mean_semantic_drift", "Mean cosine distance",
         "Semantic drift (baseline \u2192 final)", "b"),
    ]):
        vals = {t: [d[metric][t][i] for i in idx] for t in S.TOPO_ORDER}
        for j, y in enumerate(ys):
            row = [vals[t][j] for t in S.TOPO_ORDER if vals[t][j] is not None]
            if row:
                # A rule spanning the topologies makes the spread per dataset
                # legible without adding a second encoding.
                ax.hlines(y, min(row), max(row), color="#e6eaee",
                          linewidth=1.4, zorder=1)
        for t in S.TOPO_ORDER:
            ax.scatter(vals[t], ys, s=16, color=S.TOPO_COLORS[t],
                       edgecolor=S.HALO, linewidth=0.6, zorder=4)

        flat = [v for t in S.TOPO_ORDER for v in vals[t] if v is not None]
        pad = (max(flat) - min(flat)) * 0.12
        ax.set_xlim(min(flat) - pad, max(flat) + pad)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlabel(xlabel, fontsize=7.5, labelpad=2)
        ax.grid(axis="x", color=S.GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("bottom", "left"):
            ax.spines[side].set_color("#9aa5b1")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        S.panel_title(ax, title)
        S.panel_letter(ax, letter, x=-0.30 if letter == "a" else -0.06, y=1.06)

    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([S.DATASET_LABELS.get(ds, ds) for ds in order],
                            fontsize=7)
    for lab in axes[0].get_yticklabels():
        lab.set_color(S.INK)

    handles = [Line2D([], [], marker="o", linestyle="none",
                      color=S.TOPO_COLORS[t], markeredgecolor=S.HALO,
                      markeredgewidth=0.6, markersize=4.2,
                      label=S.TOPO_SHORT[t]) for t in S.TOPO_ORDER]
    leg = fig.legend(handles=handles, loc="upper right",
                     bbox_to_anchor=(0.995, 1.015), ncol=4, frameon=False,
                     fontsize=6.5, handletextpad=0.3, columnspacing=1.1)
    for txt, t in zip(leg.get_texts(), S.TOPO_ORDER):
        txt.set_color(S.TOPO_COLORS[t])
    _save(fig, "Fig9 — Semantic Mirroring and Drift by Topology")


BUILDERS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4,
            "fig5": fig5, "fig6": fig6, "fig7": fig7, "fig8": fig8,
            "fig9": fig9}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("targets", nargs="+", help="fig1 fig2 fig5 fig8, or 'all'")
    p.add_argument("--refresh", action="store_true",
                   help="recompute the data cache from outputs/ first")
    args = p.parse_args()

    targets = []
    for t in args.targets:
        if t == "all":
            targets.extend(BUILDERS)
        elif t in BUILDERS:
            targets.append(t)
        else:
            print(f"unknown target: {t}", file=sys.stderr)
            sys.exit(2)

    data = P.load_cache(refresh=args.refresh)
    S.apply_nature()
    for t in targets:
        print(f"building {t} ...")
        BUILDERS[t](data)


if __name__ == "__main__":
    main()
