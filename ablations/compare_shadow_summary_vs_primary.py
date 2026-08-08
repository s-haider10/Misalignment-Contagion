#!/usr/bin/env python3
"""Compare shadow_summary_ablation vs. primary_em on the same dataset.

Both use ratio=0.2, qwen-7b-instruct, model_induced, seed=42 — the only
difference is that shadow_summary feeds a deliberation summary to the
Stage III shadow prompt instead of the agent's own last stance/reasoning.

The comparison answers: does the shadow elicitation choice change the
measured private belief?

Outputs:
  plots_tables/shadow_summary_vs_primary/<dataset>/
    summary_table.csv
    shadow_stance_hist.png      shadow distribution: summary vs own-last
    delta_shadow_bars.png       shadow_ev - baseline_ev, both variants
    shadow_vs_final.png         shadow vs final stance per variant
    ii_srf_bars.png             internalization & shadow-reversion comparison
    pairwise_delta_hist.png     per-(trial,agent) shadow_summary - shadow_primary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from misalignment_contagion.analyze import load_trials, trials_to_dataframe

TOPOLOGIES = ["fc", "chain", "circle", "star"]
RATIO = 0.2  # shadow_summary scope


def _shadow_variant_from_path(p: str) -> str:
    return "summary" if "shadow_summary_ablation" in p else "own-last"


def load(shadow_path: str, primary_path: str) -> pd.DataFrame:
    sh = trials_to_dataframe(load_trials(shadow_path))
    pr = trials_to_dataframe(load_trials(primary_path))
    sh["variant"] = "summary"
    pr["variant"] = "own-last"
    # Keep only matching scope in primary_em
    pr = pr[
        (pr["model_condition"] == "model_induced")
        & (pr["seed"] == 42)
        & (np.isclose(pr["minority_ratio"], RATIO))
        & (pr["topology"].isin(TOPOLOGIES))
    ]
    sh = sh[np.isclose(sh["minority_ratio"], RATIO)]
    df = pd.concat([sh, pr], ignore_index=True)
    df["delta_shadow"] = df["shadow_ev"] - df["baseline_ev"]
    df["delta_final"] = df["final_ev"] - df["baseline_ev"]
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (topo, var), g in df.groupby(["topology", "variant"]):
        rows.append({
            "topology": topo,
            "variant": var,
            "n_agents": len(g),
            "baseline_ev": g["baseline_ev"].mean(),
            "final_ev": g["final_ev"].mean(),
            "shadow_ev": g["shadow_ev"].mean(),
            "shadow_stance_mean": g["shadow_stance"].dropna().mean(),
            "delta_shadow": g["delta_shadow"].mean(),
            "delta_shadow_sd": g["delta_shadow"].std(),
            "ii_mean": g["internalization_index"].dropna().mean(),
            "srf_mean": g["shadow_reversion_fraction"].dropna().mean(),
            "frac_shadow_B": float((g["shadow_stance"].dropna().astype(int) >= 5).mean()),
        })
    # pooled
    for var, g in df.groupby("variant"):
        rows.append({
            "topology": "ALL",
            "variant": var,
            "n_agents": len(g),
            "baseline_ev": g["baseline_ev"].mean(),
            "final_ev": g["final_ev"].mean(),
            "shadow_ev": g["shadow_ev"].mean(),
            "shadow_stance_mean": g["shadow_stance"].dropna().mean(),
            "delta_shadow": g["delta_shadow"].mean(),
            "delta_shadow_sd": g["delta_shadow"].std(),
            "ii_mean": g["internalization_index"].dropna().mean(),
            "srf_mean": g["shadow_reversion_fraction"].dropna().mean(),
            "frac_shadow_B": float((g["shadow_stance"].dropna().astype(int) >= 5).mean()),
        })
    return (pd.DataFrame(rows)
            .sort_values(["topology", "variant"])
            .reset_index(drop=True))


def _save(fig, name: str, out: Path) -> None:
    fig.tight_layout()
    p = out / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")


def plot_shadow_hist(df: pd.DataFrame, out: Path) -> None:
    bins = np.arange(0.5, 8.5, 1)
    variants = ["own-last", "summary"]
    fig, axes = plt.subplots(len(TOPOLOGIES), len(variants),
                             figsize=(3.2 * len(variants), 2.4 * len(TOPOLOGIES)),
                             sharex=True, sharey="row")
    for i, topo in enumerate(TOPOLOGIES):
        for j, var in enumerate(variants):
            ax = axes[i, j]
            sub = df[(df["topology"] == topo) & (df["variant"] == var)]
            vals = sub["shadow_stance"].dropna().astype(int)
            if len(vals):
                ax.hist(vals, bins=bins, edgecolor="black", alpha=0.85,
                        color="steelblue" if var == "own-last" else "salmon")
                ax.axvline(vals.mean(), color="k", lw=1.2, ls="--")
                ax.text(0.97, 0.95, f"μ={vals.mean():.2f}\nn={len(vals)}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=8)
            ax.set_xticks(range(1, 8))
            if i == 0:
                ax.set_title(f"shadow = {var}")
            if j == 0:
                ax.set_ylabel(f"{topo}\ncount")
            if i == len(TOPOLOGIES) - 1:
                ax.set_xlabel("shadow stance (1=A, 7=B)")
    fig.suptitle("Shadow stance distribution: own-last vs. summary elicitation",
                 fontsize=12, y=1.00)
    _save(fig, "shadow_stance_hist.png", out)


def plot_delta_bars(df: pd.DataFrame, out: Path) -> None:
    agg = (df.groupby(["topology", "variant"])["delta_shadow"]
             .agg(["mean", "sem"]).reset_index())
    variants = ["own-last", "summary"]
    x = np.arange(len(TOPOLOGIES))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, var in enumerate(variants):
        sub = agg[agg["variant"] == var].set_index("topology").reindex(TOPOLOGIES)
        ax.bar(x + (i - 0.5) * width, sub["mean"], width=width,
               yerr=sub["sem"], capsize=3, label=var,
               color="steelblue" if var == "own-last" else "salmon")
    ax.set_xticks(x)
    ax.set_xticklabels(TOPOLOGIES)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel("Δ shadow EV (shadow − baseline)")
    ax.set_title("Private belief shift, by shadow elicitation variant")
    ax.legend()
    _save(fig, "delta_shadow_bars.png", out)


def plot_shadow_vs_final(df: pd.DataFrame, out: Path) -> None:
    agg = (df.groupby(["variant"])
             [["final_stance", "shadow_stance"]]
             .mean().reset_index())
    variants = ["own-last", "summary"]
    x = np.arange(len(variants))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    means = agg.set_index("variant").reindex(variants)
    ax.bar(x - width/2, means["final_stance"], width=width, label="final (public)",
           color="steelblue")
    ax.bar(x + width/2, means["shadow_stance"], width=width, label="shadow (private)",
           color="salmon")
    for i, var in enumerate(variants):
        f, s = means.loc[var, "final_stance"], means.loc[var, "shadow_stance"]
        ax.text(i, max(f, s) + 0.05, f"Δ={s-f:+.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(variants)
    ax.axhline(4, color="grey", lw=0.5, ls="--")
    ax.set_ylabel("Mean stance (1=A, 7=B)")
    ax.set_title("Final vs. shadow stance, pooled topologies")
    ax.legend()
    _save(fig, "shadow_vs_final.png", out)


def plot_ii_srf(df: pd.DataFrame, out: Path) -> None:
    metrics = [("internalization_index", "Internalization Index"),
               ("shadow_reversion_fraction", "Shadow Reversion Fraction")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (col, title) in zip(axes, metrics):
        agg = (df.groupby(["topology", "variant"])[col]
                 .agg(["mean", "sem"]).reset_index())
        variants = ["own-last", "summary"]
        x = np.arange(len(TOPOLOGIES))
        width = 0.38
        for i, var in enumerate(variants):
            sub = agg[agg["variant"] == var].set_index("topology").reindex(TOPOLOGIES)
            ax.bar(x + (i - 0.5) * width, sub["mean"], width=width,
                   yerr=sub["sem"], capsize=3, label=var,
                   color="steelblue" if var == "own-last" else "salmon")
        ax.set_xticks(x); ax.set_xticklabels(TOPOLOGIES)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Private-belief metrics: own-last vs. summary elicitation", fontsize=12)
    _save(fig, "ii_srf_bars.png", out)


def plot_pairwise_delta(df: pd.DataFrame, out: Path) -> None:
    """For each (scenario, topology, position, agent_id) present in both
    variants, plot the distribution of shadow_summary_stance - own_last_stance.
    """
    key = ["scenario_id", "topology", "position_config", "agent_id"]
    summ = df[df["variant"] == "summary"].set_index(key)["shadow_stance"]
    own = df[df["variant"] == "own-last"].set_index(key)["shadow_stance"]
    common = summ.index.intersection(own.index)
    diffs = (summ.loc[common].astype(float) - own.loc[common].astype(float)).dropna()
    if not len(diffs):
        print("  no paired rows for pairwise delta plot")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(-6.5, 7.5, 1)
    ax.hist(diffs, bins=bins, edgecolor="black", color="purple", alpha=0.85)
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.axvline(diffs.mean(), color="red", lw=1.5,
               label=f"mean shift = {diffs.mean():+.2f}")
    ax.set_xlabel("shadow_summary − shadow_own-last (per agent)")
    ax.set_ylabel("count")
    ax.set_title(f"Pairwise shadow-stance shift, n={len(diffs)} matched agents")
    ax.legend()
    _save(fig, "pairwise_delta_hist.png", out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["synthetic", "moral_stories", "harmbench_standard",
                            "harmbench_contextual", "harmbench_copyright"])
    p.add_argument("--model-key", default="qwen-7b-instruct")
    p.add_argument("--out-root", default="plots_tables/shadow_summary_vs_primary")
    p.add_argument("--shadow", default=None,
                   help="Override shadow_summary path")
    p.add_argument("--primary", default=None,
                   help="Override primary_em path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    shadow_path = args.shadow or (
        f"outputs/shadow_summary_ablation/{args.dataset}/{args.model_key}/results.jsonl"
    )
    primary_path = args.primary or (
        f"outputs/primary_em/{args.dataset}/{args.model_key}/results.jsonl"
    )
    out = Path(args.out_root) / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    if not Path(shadow_path).exists():
        raise SystemExit(f"missing: {shadow_path}")
    if not Path(primary_path).exists():
        raise SystemExit(f"missing: {primary_path}")

    df = load(shadow_path, primary_path)
    print(f"Loaded {len(df)} agent rows. "
          f"variants: {sorted(df['variant'].unique())}, "
          f"topologies: {sorted(df['topology'].unique())}")
    print(df.groupby("variant")["trial_id"].nunique().to_dict())

    summary = build_summary(df)
    summary_path = out / "summary_table.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary -> {summary_path}")
    print(summary[summary["topology"] == "ALL"].to_string(index=False))
    print()
    print(summary[summary["topology"] != "ALL"].to_string(index=False))

    print("\nPlots:")
    plot_shadow_hist(df, out)
    plot_delta_bars(df, out)
    plot_shadow_vs_final(df, out)
    plot_ii_srf(df, out)
    plot_pairwise_delta(df, out)

    print(f"\nDone. Outputs in {out}/")


if __name__ == "__main__":
    main()
