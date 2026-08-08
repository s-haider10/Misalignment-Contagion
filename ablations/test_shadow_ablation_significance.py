#!/usr/bin/env python3
"""Statistical test: does the shadow elicitation method (own-last vs summary)
significantly change measured private belief?

The ablation is "successful" (in the sense of being a clean control) if the
shadow stance distribution is INVARIANT to elicitation method — i.e. p > 0.05
and small effect sizes. Significant change means the elicitation method itself
is driving the metric, which is a problem for interpreting "private belief."

Performs both paired (per-agent matched) and unpaired tests, with Cohen's d
effect size. Outputs a verdict per dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from misalignment_contagion.analyze import load_trials, trials_to_dataframe

TOPOLOGIES = ["fc", "chain", "circle", "star"]
RATIO = 0.2


def load(shadow_path: str, primary_path: str) -> pd.DataFrame:
    sh = trials_to_dataframe(load_trials(shadow_path))
    pr = trials_to_dataframe(load_trials(primary_path))
    sh["variant"] = "summary"
    pr["variant"] = "own-last"
    pr = pr[
        (pr["model_condition"] == "model_induced")
        & (pr["seed"] == 42)
        & (np.isclose(pr["minority_ratio"], RATIO))
        & (pr["topology"].isin(TOPOLOGIES))
    ]
    sh = sh[np.isclose(sh["minority_ratio"], RATIO)]
    df = pd.concat([sh, pr], ignore_index=True)
    df["delta_shadow"] = df["shadow_ev"] - df["baseline_ev"]
    return df


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                / (len(a) + len(b) - 2))
    if s == 0:
        return float("nan")
    return (a.mean() - b.mean()) / s


def paired_test(df: pd.DataFrame, col: str) -> dict:
    """Wilcoxon signed-rank on matched (scenario, topology, position, agent)."""
    key = ["scenario_id", "topology", "position_config", "agent_id"]
    summ = df[df["variant"] == "summary"].set_index(key)[col]
    own = df[df["variant"] == "own-last"].set_index(key)[col]
    common = summ.index.intersection(own.index)
    s = summ.loc[common].astype(float)
    o = own.loc[common].astype(float)
    valid = (~s.isna()) & (~o.isna())
    s, o = s[valid].values, o[valid].values
    if len(s) < 10:
        return {"n_pairs": len(s), "wilcoxon_p": float("nan"),
                "mean_diff": float("nan"), "cohen_d_paired": float("nan")}
    diff = s - o
    if np.allclose(diff, 0):
        return {"n_pairs": len(s), "wilcoxon_p": 1.0,
                "mean_diff": 0.0, "cohen_d_paired": 0.0}
    try:
        stat, p = stats.wilcoxon(diff)
    except ValueError:
        p = float("nan")
    return {
        "n_pairs": int(len(s)),
        "wilcoxon_p": float(p),
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "cohen_d_paired": float(diff.mean() / diff.std(ddof=1))
                          if diff.std(ddof=1) > 0 else 0.0,
    }


def unpaired_test(df: pd.DataFrame, col: str) -> dict:
    s = df[df["variant"] == "summary"][col].dropna().astype(float).values
    o = df[df["variant"] == "own-last"][col].dropna().astype(float).values
    if len(s) < 10 or len(o) < 10:
        return {"n_summary": len(s), "n_own": len(o),
                "mwu_p": float("nan"), "cohen_d": float("nan")}
    stat, p = stats.mannwhitneyu(s, o, alternative="two-sided")
    return {
        "n_summary": int(len(s)),
        "n_own": int(len(o)),
        "mean_summary": float(s.mean()),
        "mean_own": float(o.mean()),
        "mwu_p": float(p),
        "cohen_d": cohens_d(s, o),
    }


def verdict(p: float, d: float) -> str:
    """Interpret p + effect size for an invariance ablation."""
    if np.isnan(p) or np.isnan(d):
        return "INSUFFICIENT DATA"
    ad = abs(d)
    if p > 0.05:
        return "✓ INVARIANT (ablation control passes)"
    # significant
    if ad < 0.1:
        return "⚠ significant but TRIVIAL effect (p<0.05, |d|<0.1)"
    if ad < 0.2:
        return "⚠ SMALL effect (Cohen's d ~ 0.1-0.2) — elicitation slightly matters"
    if ad < 0.5:
        return "✗ MEDIUM effect — elicitation method materially changes result"
    return "✗ LARGE effect — elicitation method dominates the measurement"


def analyze_dataset(dataset: str, model_key: str = "qwen-7b-instruct") -> None:
    shadow_path = f"outputs/shadow_summary_ablation/{dataset}/{model_key}/results.jsonl"
    primary_path = f"outputs/primary_em/{dataset}/{model_key}/results.jsonl"
    if not Path(shadow_path).exists() or not Path(primary_path).exists():
        print(f"\n=== {dataset}: SKIP (missing data) ===")
        return

    df = load(shadow_path, primary_path)
    n_sh = df[df["variant"] == "summary"]["trial_id"].nunique()
    n_pr = df[df["variant"] == "own-last"]["trial_id"].nunique()
    print(f"\n{'='*72}")
    print(f"=== {dataset}  (summary trials: {n_sh}, own-last trials: {n_pr}) ===")
    print(f"{'='*72}")

    for col, name in [
        ("shadow_stance", "shadow_stance (discrete 1-7)"),
        ("shadow_ev",     "shadow_ev (logprob expectation)"),
        ("delta_shadow",  "delta_shadow (shadow_ev - baseline_ev)"),
    ]:
        print(f"\n  Metric: {name}")
        pair = paired_test(df, col)
        unpr = unpaired_test(df, col)
        print(f"    Paired Wilcoxon  : n={pair['n_pairs']}, "
              f"mean diff={pair['mean_diff']:+.3f}, "
              f"p={pair['wilcoxon_p']:.2e}, d_paired={pair['cohen_d_paired']:+.3f}")
        print(f"    Unpaired Mann-Whitney : n_sum={unpr['n_summary']}, "
              f"n_own={unpr['n_own']}, "
              f"mean(sum)={unpr['mean_summary']:.3f}, "
              f"mean(own)={unpr['mean_own']:.3f}, "
              f"p={unpr['mwu_p']:.2e}, d={unpr['cohen_d']:+.3f}")
        v_pair = verdict(pair['wilcoxon_p'], pair['cohen_d_paired'])
        v_unpr = verdict(unpr['mwu_p'], unpr['cohen_d'])
        print(f"    Verdict (paired)   : {v_pair}")
        print(f"    Verdict (unpaired) : {v_unpr}")


def analyze_pooled(datasets: list[str], model_key: str = "qwen-7b-instruct") -> None:
    """Pool all datasets together and report ONE verdict per metric.

    For the invariance question, the right test on the pooled data is:
    'across all data we have, does elicitation method change shadow stance?'
    Sign of the effect can flip per dataset, so we report two views:
      (a) raw pooled difference (signed) — what's the net effect?
      (b) pooled absolute paired difference — how big is the perturbation
          regardless of direction?
    """
    frames = []
    for ds in datasets:
        sh = f"outputs/shadow_summary_ablation/{ds}/{model_key}/results.jsonl"
        pr = f"outputs/primary_em/{ds}/{model_key}/results.jsonl"
        if not Path(sh).exists() or not Path(pr).exists():
            continue
        df = load(sh, pr)
        df["dataset"] = ds
        frames.append(df)
    if not frames:
        print("\nNo data to pool.")
        return
    pool = pd.concat(frames, ignore_index=True)

    n_sh = pool[pool["variant"] == "summary"]["trial_id"].nunique()
    n_pr = pool[pool["variant"] == "own-last"]["trial_id"].nunique()
    print(f"\n{'='*72}")
    print(f"=== POOLED ACROSS DATASETS  "
          f"(summary trials: {n_sh}, own-last trials: {n_pr}) ===")
    print(f"=== datasets included: {sorted(pool['dataset'].unique())} ===")
    print(f"{'='*72}")

    for col, name in [
        ("shadow_stance", "shadow_stance (discrete 1-7)"),
        ("shadow_ev",     "shadow_ev (logprob expectation)"),
        ("delta_shadow",  "delta_shadow (shadow_ev - baseline_ev)"),
    ]:
        # Paired: per (dataset, scenario, topology, position, agent)
        key = ["dataset", "scenario_id", "topology", "position_config", "agent_id"]
        summ = pool[pool["variant"] == "summary"].set_index(key)[col]
        own = pool[pool["variant"] == "own-last"].set_index(key)[col]
        common = summ.index.intersection(own.index)
        s = summ.loc[common].astype(float)
        o = own.loc[common].astype(float)
        valid = (~s.isna()) & (~o.isna())
        s, o = s[valid].values, o[valid].values
        diff = s - o
        abs_diff = np.abs(diff)

        # Signed test
        try:
            _, p_signed = stats.wilcoxon(diff)
        except ValueError:
            p_signed = float("nan")
        d_signed = (diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0

        # Magnitude test: is |diff| meaningfully > 0? (one-sided)
        # Compare |diff| to 0 with Wilcoxon signed-rank, alt='greater'.
        try:
            _, p_mag = stats.wilcoxon(abs_diff, alternative="greater")
        except ValueError:
            p_mag = float("nan")

        # Per-dataset signed mean diffs to show the flip
        per_ds = (pool.set_index(key)
                  .pivot_table(index="dataset", columns="variant",
                               values=col, aggfunc="mean")
                  .assign(mean_diff=lambda d: d.get("summary", 0) - d.get("own-last", 0)))

        print(f"\n  Metric: {name}")
        print(f"    n matched pairs: {len(diff)}")
        print(f"    Signed mean diff (summary - own-last) : {diff.mean():+.4f}")
        print(f"    Mean ABSOLUTE diff (|summary - own-last|): {abs_diff.mean():.4f}")
        print(f"    Median diff                              : {np.median(diff):+.4f}")
        print(f"    Cohen's d (paired, signed)               : {d_signed:+.3f}")
        print(f"    Wilcoxon (signed, two-sided) p           : {p_signed:.2e}")
        print(f"    Wilcoxon (|diff| > 0, one-sided) p       : {p_mag:.2e}")

        print(f"    Per-dataset signed mean diff:")
        for ds_name, row in per_ds.iterrows():
            md = row.get("mean_diff", float("nan"))
            print(f"      {ds_name:25s} {md:+.4f}")

        print(f"    Verdict (signed, mean effect): "
              f"{verdict(p_signed, d_signed)}")
        # Magnitude verdict: |d| analogue using mean |diff| / sd of |diff|
        sd_abs = abs_diff.std(ddof=1)
        d_mag = (abs_diff.mean() / sd_abs) if sd_abs > 0 else 0.0
        # For one-sided magnitude, "invariant" => p > 0.05; effect size on
        # raw absolute differences relative to their spread:
        if np.isnan(p_mag):
            v_mag = "INSUFFICIENT DATA"
        elif p_mag > 0.05:
            v_mag = "✓ no detectable perturbation magnitude"
        else:
            v_mag = (f"✗ elicitation perturbs shadow by mean "
                     f"|{abs_diff.mean():.2f}| stance units (p={p_mag:.1e})")
        print(f"    Verdict (magnitude)         : {v_mag}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["synthetic", "moral_stories", "harmbench_standard"])
    p.add_argument("--pooled-only", action="store_true",
                   help="Skip per-dataset breakdown, just print pooled verdict.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pooled_only:
        for ds in args.datasets:
            analyze_dataset(ds)
    analyze_pooled(args.datasets)
    print("\n" + "=" * 72)
    print("KEY: paired tests use per-(dataset,scenario,topology,position,agent) matches.")
    print("     |Cohen's d| ~ 0.2 small, 0.5 medium, 0.8 large.")
    print("     'Signed' tests whether means differ; sign can cancel across datasets.")
    print("     'Magnitude' tests whether |diff| > 0; cannot cancel.")
    print("=" * 72)


if __name__ == "__main__":
    main()
