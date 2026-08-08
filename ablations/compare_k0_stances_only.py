#!/usr/bin/env python3
"""Focused k0-vs-primary_em comparison: final and shadow stance only.

For aligned agents, plots histograms and means of:
  - final_stance  (last deliberation round)
  - shadow_stance (private post-deliberation)
across k=0, k=1, k=2, k=3, per topology and pooled.

Outputs:
  plots_tables/k0_stances/
    stance_summary.csv
    final_stance_hist.png       histogram per condition, faceted by topology
    shadow_stance_hist.png      same for shadow
    mean_stance_bars.png        mean final/shadow per (topology, condition)
    final_vs_shadow_scatter.png mean shift between final and shadow, per condition
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from misalignment_contagion.analyze import load_trials, trials_to_dataframe

K0 = "outputs/k0_baseline/synthetic/qwen-7b-instruct/results.jsonl"
PRIMARY = "outputs/primary_em/synthetic/qwen-7b-instruct/results.jsonl"
OUT = "plots_tables/k0_stances"

TOPOLOGIES = ["fc", "chain", "circle", "star"]
CONDITION_ORDER = ["k=0", "k=1", "k=2", "k=3"]


def _cond(r: float) -> str:
    return f"k={int(round(r * 10))}"


def load(k0_path: str, primary_path: str) -> pd.DataFrame:
    k0 = trials_to_dataframe(load_trials(k0_path))
    pr = trials_to_dataframe(load_trials(primary_path))
    pr = pr[
        (pr["model_condition"] == "model_induced")
        & (pr["seed"] == 42)
        & (pr["position_config"] == 0)
        & (pr["topology"].isin(TOPOLOGIES))
    ]
    df = pd.concat([k0, pr], ignore_index=True)
    df["condition"] = df["minority_ratio"].apply(_cond)
    return df[["condition", "topology", "minority_ratio", "final_stance", "shadow_stance"]]


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (topo, cond), g in df.groupby(["topology", "condition"]):
        for stage, col in [("final", "final_stance"), ("shadow", "shadow_stance")]:
            vals = g[col].dropna().astype(int)
            rows.append({
                "topology": topo,
                "condition": cond,
                "stage": stage,
                "n": len(vals),
                "mean": vals.mean(),
                "median": vals.median(),
                "std": vals.std(),
                "frac_A_side": float((vals <= 3).mean()),
                "frac_neutral": float((vals == 4).mean()),
                "frac_B_side": float((vals >= 5).mean()),
            })
    # Pooled rows (all topologies)
    for cond, g in df.groupby("condition"):
        for stage, col in [("final", "final_stance"), ("shadow", "shadow_stance")]:
            vals = g[col].dropna().astype(int)
            rows.append({
                "topology": "ALL",
                "condition": cond,
                "stage": stage,
                "n": len(vals),
                "mean": vals.mean(),
                "median": vals.median(),
                "std": vals.std(),
                "frac_A_side": float((vals <= 3).mean()),
                "frac_neutral": float((vals == 4).mean()),
                "frac_B_side": float((vals >= 5).mean()),
            })
    out = pd.DataFrame(rows)
    out["cond_order"] = out["condition"].map({c: i for i, c in enumerate(CONDITION_ORDER)})
    return out.sort_values(["topology", "stage", "cond_order"]).drop(columns="cond_order").reset_index(drop=True)


def _save(fig, name: str, out: Path) -> None:
    fig.tight_layout()
    p = out / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")


def plot_stance_hist(df: pd.DataFrame, out: Path, col: str, fname: str, title: str) -> None:
    bins = np.arange(0.5, 8.5, 1)
    fig, axes = plt.subplots(len(TOPOLOGIES), len(CONDITION_ORDER),
                             figsize=(3.0 * len(CONDITION_ORDER), 2.4 * len(TOPOLOGIES)),
                             sharex=True, sharey="row")
    for i, topo in enumerate(TOPOLOGIES):
        for j, cond in enumerate(CONDITION_ORDER):
            ax = axes[i, j]
            sub = df[(df["topology"] == topo) & (df["condition"] == cond)]
            vals = sub[col].dropna().astype(int)
            if len(vals):
                ax.hist(vals, bins=bins, edgecolor="black", alpha=0.85,
                        color="steelblue" if cond == "k=0" else "salmon")
                mean = vals.mean()
                ax.axvline(mean, color="black", lw=1.2, ls="--")
                ax.text(0.97, 0.95, f"μ={mean:.2f}\nn={len(vals)}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=8)
            ax.set_xticks(range(1, 8))
            if i == 0:
                ax.set_title(cond)
            if j == 0:
                ax.set_ylabel(f"{topo}\ncount")
            if i == len(TOPOLOGIES) - 1:
                ax.set_xlabel("stance (1=A, 7=B)")
    fig.suptitle(title, fontsize=13, y=1.00)
    _save(fig, fname, out)


def plot_mean_stance_bars(df: pd.DataFrame, out: Path) -> None:
    rows = []
    for (topo, cond), g in df.groupby(["topology", "condition"]):
        rows.append({
            "topology": topo, "condition": cond,
            "final_mean": g["final_stance"].dropna().mean(),
            "final_sem": g["final_stance"].dropna().sem(),
            "shadow_mean": g["shadow_stance"].dropna().mean(),
            "shadow_sem": g["shadow_stance"].dropna().sem(),
        })
    agg = pd.DataFrame(rows)
    x = np.arange(len(TOPOLOGIES))
    width = 0.20
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, stage in zip(axes, ["final", "shadow"]):
        for i, cond in enumerate(CONDITION_ORDER):
            sub = agg[agg["condition"] == cond].set_index("topology").reindex(TOPOLOGIES)
            ax.bar(x + (i - 1.5) * width, sub[f"{stage}_mean"],
                   width=width, yerr=sub[f"{stage}_sem"], capsize=3, label=cond)
        ax.set_xticks(x)
        ax.set_xticklabels(TOPOLOGIES)
        ax.set_title(f"mean {stage} stance")
        ax.set_xlabel("topology")
        ax.axhline(4, color="grey", lw=0.5, ls="--")
        ax.set_ylabel("stance (1=A ··· 7=B)")
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("Mean final vs. shadow stance, by topology and condition", fontsize=12)
    _save(fig, "mean_stance_bars.png", out)


def plot_final_vs_shadow(df: pd.DataFrame, out: Path) -> None:
    rows = []
    for cond in CONDITION_ORDER:
        sub = df[df["condition"] == cond]
        rows.append({
            "condition": cond,
            "final_mean": sub["final_stance"].dropna().mean(),
            "shadow_mean": sub["shadow_stance"].dropna().mean(),
            "shift": sub["shadow_stance"].dropna().mean()
                     - sub["final_stance"].dropna().mean(),
        })
    agg = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(agg))
    width = 0.35
    ax.bar(x - width / 2, agg["final_mean"], width=width, label="final (public)",
           color="steelblue")
    ax.bar(x + width / 2, agg["shadow_mean"], width=width, label="shadow (private)",
           color="salmon")
    for i, row in agg.iterrows():
        ax.text(i, max(row["final_mean"], row["shadow_mean"]) + 0.05,
                f"Δ={row['shift']:+.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["condition"])
    ax.set_ylabel("mean stance (1=A ··· 7=B)")
    ax.set_title("Final vs. shadow stance per condition (pooled topologies)")
    ax.axhline(4, color="grey", lw=0.5, ls="--")
    ax.legend()
    _save(fig, "final_vs_shadow_scatter.png", out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--k0", default=K0)
    p.add_argument("--primary", default=PRIMARY)
    p.add_argument("--out", default=OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load(args.k0, args.primary)
    print(f"Loaded {len(df)} aligned-agent rows. "
          f"Conditions: {sorted(df['condition'].unique())}, "
          f"Topologies: {sorted(df['topology'].unique())}")

    summary = build_summary(df)
    summary_path = out / "stance_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary -> {summary_path}")

    pooled = summary[summary["topology"] == "ALL"].drop(columns="topology")
    print("\n== Pooled (all topologies) ==")
    print(pooled.to_string(index=False))

    print("\nPlots:")
    plot_stance_hist(df, out, "final_stance",
                     "final_stance_hist.png",
                     "Final (public) stance distribution — aligned agents")
    plot_stance_hist(df, out, "shadow_stance",
                     "shadow_stance_hist.png",
                     "Shadow (private) stance distribution — aligned agents")
    plot_mean_stance_bars(df, out)
    plot_final_vs_shadow(df, out)

    print(f"\nDone. Outputs in {out}/")


if __name__ == "__main__":
    main()
