"""Shared template for the three shadow ablation figures.

Each shadow ablation compares one alternative Stage III elicitation design
against the primary "own-last" condition. Headline panel: II and SRF (the
two persistence metrics) per dataset, primary vs ablation side by side.
Caption notes whether differences are within sampling noise.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..fig_style import (apply, style_axes, add_zero_rule, panel_label,
                          DATASET_COLORS, DATASET_LABELS, DATASET_ORDER,
                          MUTED, GRID, RULE)
from ._common import PLOTS_TABLES, ensure_out

warnings.filterwarnings("ignore")


def _load_pooled(ablation_dir: str, primary_variant: str,
                 abl_variant: str) -> pd.DataFrame:
    """Read every dataset's summary_table.csv and pool the ALL-topology row.

    Returns one row per (dataset, variant) with ii_mean, srf_mean,
    delta_shadow, n_agents.
    """
    root = PLOTS_TABLES / ablation_dir
    rows = []
    for ds in DATASET_ORDER:
        f = root / ds / "summary_table.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        sub = df[df["topology"] == "ALL"]
        for _, r in sub.iterrows():
            v = r["variant"]
            if v not in (primary_variant, abl_variant):
                continue
            rows.append({
                "dataset": ds,
                "variant": v,
                "ii_mean": float(r["ii_mean"]),
                "srf_mean": float(r["srf_mean"]),
                "delta_shadow": float(r["delta_shadow"]),
                "shadow_stance_mean": float(r["shadow_stance_mean"]),
                "n_agents": int(r["n_agents"]),
            })
    return pd.DataFrame(rows)


def _grouped_bars(ax, df, value_col, primary_variant, abl_variant,
                  primary_label, abl_label, ylabel, title,
                  ylim=None, clip=None):
    """Plot grouped bars: x=dataset, group=(primary, ablation).

    If `clip=(lo, hi)` is given, values outside this range are clipped and a
    tiny annotation marks the bar as clipped (II/SRF can blow up when the
    public deliberation shift is near zero, which happens on harmbench).
    """
    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    x = np.arange(len(datasets))
    w = 0.36

    def _get(d, variant):
        sub = df[(df["dataset"] == d) & (df["variant"] == variant)]
        if sub.empty:
            return np.nan, False
        v = float(sub[value_col].iloc[0])
        clipped = False
        if clip is not None and np.isfinite(v):
            lo, hi = clip
            if v < lo:
                v, clipped = lo, True
            elif v > hi:
                v, clipped = hi, True
        return v, clipped

    prim = [_get(d, primary_variant) for d in datasets]
    abl = [_get(d, abl_variant) for d in datasets]
    prim_vals = [p[0] for p in prim]
    abl_vals = [a[0] for a in abl]
    colors = [DATASET_COLORS[d] for d in datasets]

    ax.bar(x - w / 2, prim_vals, width=w,
           color=colors, alpha=0.55, hatch="//",
           edgecolor="white", linewidth=0.6, label=primary_label, zorder=3)
    ax.bar(x + w / 2, abl_vals, width=w,
           color=colors, alpha=1.0,
           edgecolor="white", linewidth=0.6, label=abl_label, zorder=3)

    # Mark clipped bars with a small "*" cap
    if clip is not None:
        for i, ((v, c), (av, ac)) in enumerate(zip(prim, abl)):
            if c:
                ax.text(x[i] - w / 2, v, "*", ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=10, color=MUTED, zorder=4)
            if ac:
                ax.text(x[i] + w / 2, av, "*", ha="center",
                        va="bottom" if av >= 0 else "top",
                        fontsize=10, color=MUTED, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets],
                        rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, pad=6)
    add_zero_rule(ax, axis="y")
    style_axes(ax, grid_axis="y")
    if ylim is not None:
        ax.set_ylim(*ylim)


def build_shadow_figure(ablation_dir: str,
                        primary_variant: str,
                        abl_variant: str,
                        primary_label: str,
                        abl_label: str,
                        figure_title: str,
                        out_stem: str,
                        caption: str):
    apply()
    out_dir = ensure_out()
    df = _load_pooled(ablation_dir, primary_variant, abl_variant)
    if df.empty:
        print(f"  (skip) {out_stem}: no data under {ablation_dir}")
        return None

    df.to_csv(out_dir / f"{out_stem}_data.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    panel_label(axes[0], "A", x=-0.22, y=1.07)
    panel_label(axes[1], "B", x=-0.16, y=1.07)
    panel_label(axes[2], "C", x=-0.16, y=1.07)

    _grouped_bars(axes[0], df, "ii_mean", primary_variant, abl_variant,
                   primary_label, abl_label,
                   "Mean II (persistence)",
                   "Internalization Index",
                   ylim=(0, 3.2), clip=(0, 3.0))
    _grouped_bars(axes[1], df, "srf_mean", primary_variant, abl_variant,
                   primary_label, abl_label,
                   "Mean SRF",
                   "Shadow Reversion Fraction",
                   ylim=(-2.2, 2.2), clip=(-2.0, 2.0))
    _grouped_bars(axes[2], df, "delta_shadow", primary_variant, abl_variant,
                   primary_label, abl_label,
                   r"$\Delta$shadow (shadow$-$baseline EV)",
                   "Public-private shift",
                   ylim=None)

    # Single legend for the whole figure
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#999999", alpha=0.55,
                       hatch="//", edgecolor="white", linewidth=0.6,
                       label=primary_label),
        plt.Rectangle((0, 0), 1, 1, facecolor="#999999", alpha=1.0,
                       edgecolor="white", linewidth=0.6,
                       label=abl_label),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
                bbox_to_anchor=(0.5, -0.04), fontsize=9, frameon=False)
    fig.suptitle(figure_title, fontsize=12, fontweight="bold", y=1.02)
    full_caption = (caption + "  * = value clipped to display range "
                    "(II∈[0,3], SRF∈[-2,2]); occurs on HarmBench cells where "
                    "tiny public shifts inflate the ratio.")
    fig.text(0.5, -0.10, full_caption, ha="center", va="top",
             fontsize=8.5, color=MUTED, style="italic", wrap=True)

    fig.tight_layout()
    png = out_dir / f"{out_stem}.png"
    pdf = out_dir / f"{out_stem}.pdf"
    fig.savefig(png, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png.relative_to(png.parents[2])}")
    return df
