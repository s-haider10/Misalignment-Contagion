#!/usr/bin/env python3
"""Compare the k0_baseline (all-aligned, ratio=0) run against the matched
primary_em condition (qwen-7b-instruct, model_induced, synthetic, seed=42)
and plot key differences.

Outputs:
  plots_tables/k0_vs_primary/
    summary_table.csv          per-(topology x ratio) aggregate metrics
    delta_ev_bars.png          shift = final_ev - baseline_ev, public
    delta_shadow_bars.png      shift = shadow_ev - baseline_ev, private
    ev_trajectory.png          EV across rounds (baseline, R0..R4, shadow)
    entropy_trajectory.png     Shannon entropy across stages
    conversion_rate_bars.png   fraction of aligned agents who flipped stance
    final_stance_hist.png      histogram of final stances by condition
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from misalignment_contagion.analyze import load_trials, trials_to_dataframe

K0_DEFAULT = "outputs/k0_baseline/synthetic/qwen-7b-instruct/results.jsonl"
PRIMARY_DEFAULT = "outputs/primary_em/synthetic/qwen-7b-instruct/results.jsonl"
OUT_DEFAULT = "plots_tables/k0_vs_primary"

TOPOLOGIES = ["fc", "chain", "circle", "star"]


# ── Helpers ──────────────────────────────────────────────────────────

def _flipped(row: pd.Series) -> bool:
    """An aligned agent has 'flipped' if its final stance crossed the midpoint
    (≤3 → ≥5 or ≥5 → ≤3) relative to baseline. 4 is treated as no-flip."""
    b = row["baseline_stance"]
    f = row["final_stance"]
    if b is None or f is None:
        return False
    if b <= 3 and f >= 5:
        return True
    if b >= 5 and f <= 3:
        return True
    return False


def _ev_traj(row: pd.Series) -> list[float]:
    """Return [baseline_ev, *round_evs, shadow_ev] with None filtered as NaN."""
    pts = [row["baseline_ev"]] + list(row["round_evs"]) + [row["shadow_ev"]]
    return [np.nan if p is None else p for p in pts]


def _entropy_traj(row: pd.Series) -> list[float]:
    return [np.nan if p is None else p for p in row["entropy_trajectory"]]


def load_and_filter(k0_path: str, primary_path: str) -> pd.DataFrame:
    k0 = trials_to_dataframe(load_trials(k0_path))
    pr = trials_to_dataframe(load_trials(primary_path))
    pr = pr[
        (pr["model_condition"] == "model_induced")
        & (pr["seed"] == 42)
        & (pr["position_config"] == 0)
        & (pr["topology"].isin(TOPOLOGIES))
    ]
    df = pd.concat([k0, pr], ignore_index=True)
    df["condition"] = df["minority_ratio"].apply(
        lambda r: "k0 (no minority)" if r == 0.0 else f"k={int(round(r*10))} (ratio={r})"
    )
    df["delta_ev"] = df["final_ev"] - df["baseline_ev"]
    df["delta_shadow"] = df["shadow_ev"] - df["baseline_ev"]
    df["flipped"] = df.apply(_flipped, axis=1)
    return df


# ── Tables ───────────────────────────────────────────────────────────

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["topology", "condition", "minority_ratio"], as_index=False).agg(
        n_agents=("trial_id", "count"),
        baseline_ev=("baseline_ev", "mean"),
        final_ev=("final_ev", "mean"),
        shadow_ev=("shadow_ev", "mean"),
        delta_ev=("delta_ev", "mean"),
        delta_shadow=("delta_shadow", "mean"),
        delta_ev_sd=("delta_ev", "std"),
        delta_shadow_sd=("delta_shadow", "std"),
        conversion_rate=("flipped", "mean"),
    )
    g = g.sort_values(["topology", "minority_ratio"]).reset_index(drop=True)
    return g


# ── Plots ────────────────────────────────────────────────────────────

def _save(fig, name: str, out: Path) -> None:
    fig.tight_layout()
    p = out / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")


def plot_delta_bars(df: pd.DataFrame, out: Path, col: str, ylabel: str, fname: str) -> None:
    agg = df.groupby(["topology", "condition"])[col].agg(["mean", "sem"]).reset_index()
    conditions = sorted(agg["condition"].unique())
    topologies = TOPOLOGIES
    x = np.arange(len(topologies))
    width = 0.8 / max(len(conditions), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cond in enumerate(conditions):
        sub = agg[agg["condition"] == cond].set_index("topology").reindex(topologies)
        ax.bar(x + i * width - 0.4 + width / 2, sub["mean"],
               width=width, yerr=sub["sem"], capsize=3, label=cond)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(topologies)
    ax.set_xlabel("Topology")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by topology and condition\n(k=0 vs. primary_em)")
    ax.legend(loc="best", fontsize=9)
    _save(fig, fname, out)


def plot_ev_trajectory(df: pd.DataFrame, out: Path) -> None:
    df = df.copy()
    df["ev_traj"] = df.apply(_ev_traj, axis=1)
    stage_labels = ["baseline", "R0", "R1", "R2", "R3", "R4", "shadow"]
    conditions = sorted(df["condition"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, topo in zip(axes.flat, TOPOLOGIES):
        sub = df[df["topology"] == topo]
        for cond in conditions:
            ssub = sub[sub["condition"] == cond]
            if ssub.empty:
                continue
            arr = np.array([row for row in ssub["ev_traj"]], dtype=float)
            mean = np.nanmean(arr, axis=0)
            sem = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
            ax.plot(stage_labels, mean, marker="o", label=cond)
            ax.fill_between(stage_labels, mean - sem, mean + sem, alpha=0.15)
        ax.set_title(topo)
        ax.set_ylabel("Mean stance EV (1=A, 7=B)")
        ax.axhline(4, color="grey", lw=0.5, ls="--")
        ax.tick_params(axis="x", rotation=30)
    axes.flat[0].legend(loc="best", fontsize=8)
    fig.suptitle("Stance EV trajectory: k=0 vs. primary_em (qwen-7b-instruct, synthetic)",
                 fontsize=12)
    _save(fig, "ev_trajectory.png", out)


def plot_entropy_trajectory(df: pd.DataFrame, out: Path) -> None:
    df = df.copy()
    df["ent_traj"] = df.apply(_entropy_traj, axis=1)
    stage_labels = ["baseline", "R0", "R1", "R2", "R3", "R4", "shadow"]
    conditions = sorted(df["condition"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, topo in zip(axes.flat, TOPOLOGIES):
        sub = df[df["topology"] == topo]
        for cond in conditions:
            ssub = sub[sub["condition"] == cond]
            if ssub.empty:
                continue
            arr = np.array([row for row in ssub["ent_traj"]], dtype=float)
            mean = np.nanmean(arr, axis=0)
            sem = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
            ax.plot(stage_labels, mean, marker="o", label=cond)
            ax.fill_between(stage_labels, mean - sem, mean + sem, alpha=0.15)
        ax.set_title(topo)
        ax.set_ylabel("Shannon entropy")
        ax.tick_params(axis="x", rotation=30)
    axes.flat[0].legend(loc="best", fontsize=8)
    fig.suptitle("Belief entropy trajectory: k=0 vs. primary_em",
                 fontsize=12)
    _save(fig, "entropy_trajectory.png", out)


def plot_conversion_rate(df: pd.DataFrame, out: Path) -> None:
    agg = df.groupby(["topology", "condition"])["flipped"].mean().reset_index()
    conditions = sorted(agg["condition"].unique())
    x = np.arange(len(TOPOLOGIES))
    width = 0.8 / max(len(conditions), 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cond in enumerate(conditions):
        sub = agg[agg["condition"] == cond].set_index("topology").reindex(TOPOLOGIES)
        ax.bar(x + i * width - 0.4 + width / 2, sub["flipped"],
               width=width, label=cond)
    ax.set_xticks(x)
    ax.set_xticklabels(TOPOLOGIES)
    ax.set_ylabel("Conversion rate (fraction of aligned agents who flipped sides)")
    ax.set_title("Conversion rate by topology and condition")
    ax.legend(loc="best", fontsize=9)
    _save(fig, "conversion_rate_bars.png", out)


def plot_final_stance_hist(df: pd.DataFrame, out: Path) -> None:
    conditions = sorted(df["condition"].unique())
    fig, axes = plt.subplots(1, len(conditions), figsize=(4 * len(conditions), 4),
                             sharey=True)
    if len(conditions) == 1:
        axes = [axes]
    bins = np.arange(0.5, 8.5, 1)
    for ax, cond in zip(axes, conditions):
        sub = df[df["condition"] == cond]
        vals = sub["final_stance"].dropna().astype(int)
        ax.hist(vals, bins=bins, edgecolor="black", alpha=0.8)
        ax.set_xticks(range(1, 8))
        ax.set_title(f"{cond}\n(n={len(vals)})")
        ax.set_xlabel("Final stance (1=A, 7=B)")
    axes[0].set_ylabel("Count of aligned agents")
    fig.suptitle("Final-round stance distribution (aligned agents only)", fontsize=12)
    _save(fig, "final_stance_hist.png", out)


# ── Main ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--k0", default=K0_DEFAULT)
    p.add_argument("--primary", default=PRIMARY_DEFAULT)
    p.add_argument("--out", default=OUT_DEFAULT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_and_filter(args.k0, args.primary)
    n_trials = df["trial_id"].nunique()
    n_agents = len(df)
    print(f"Loaded {n_trials} trials, {n_agents} aligned-agent rows.")
    print("Conditions:", sorted(df["condition"].unique()))
    print("Topologies:", sorted(df["topology"].unique()))

    summary = build_summary(df)
    summary_path = out / "summary_table.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary table -> {summary_path}")
    print(summary.to_string(index=False))

    print("\nPlots:")
    plot_delta_bars(df, out, "delta_ev",
                    "Δ public EV (final − baseline)",
                    "delta_ev_bars.png")
    plot_delta_bars(df, out, "delta_shadow",
                    "Δ private EV (shadow − baseline)",
                    "delta_shadow_bars.png")
    plot_ev_trajectory(df, out)
    plot_entropy_trajectory(df, out)
    plot_conversion_rate(df, out)
    plot_final_stance_hist(df, out)

    print(f"\nDone. Outputs in {out}/")


if __name__ == "__main__":
    main()
