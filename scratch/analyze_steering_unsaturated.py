"""Re-analyze v1 restricted to non-saturated agent-rows.

Hypothesis: the +0.166 EV contrast in v1 is averaging across many
saturated agent-rows (max_p=1.0 in both conditions, contributing 0 to
any difference) and a smaller subset of non-saturated rows where the
real causal effect lives.

If we restrict to agent-rows where the model still has room to move
(max_p < 0.99 at round_4 in at least one condition), the effect
size should be larger.

We compute:
  - paired t-test on round_4 EV restricted to non-saturated cases
  - sign analysis on (pos_all - neg_all) restricted to same
  - comparison of effect size with/without restriction
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

JSONL = Path("outputs/steering_experiment/trials.jsonl")
SAT_THRESHOLD = 0.99


def ev(p):
    if not p:
        return float("nan")
    return sum(int(k) * float(v) for k, v in p.items())


def maxp(p):
    if not p:
        return float("nan")
    return max(p.values())


# Load all agent-row data
rows = []
with open(JSONL) as f:
    for line in f:
        if not line.strip():
            continue
        tr = json.loads(line)
        for a in tr["agents"]:
            r4_probs = a["round_probs"][-1]
            rows.append({
                "scenario_id": tr["scenario_id"],
                "condition": tr["condition"],
                "agent_id": a["agent_id"],
                "round_4_ev": ev(r4_probs),
                "round_4_maxp": maxp(r4_probs),
            })

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} agent rows from {df['scenario_id'].nunique()} scenarios "
      f"× {df['condition'].nunique()} conditions\n")


def paired_test(metric_filter_label, df_filtered):
    """Run paired tests on filtered data."""
    # Group by scenario + agent_id to get within-agent pairs
    pivot = df_filtered.pivot_table(
        index=["scenario_id", "agent_id"],
        columns="condition",
        values="round_4_ev",
    )
    print(f"\n=== {metric_filter_label} ===")
    print(f"  agent-rows after filter: {len(df_filtered)}, "
          f"agent-pairs available: {len(pivot)}")

    for cond in ["pos_all", "neg_all", "pos_r2"]:
        if cond not in pivot.columns or "control" not in pivot.columns:
            continue
        diff = (pivot[cond] - pivot["control"]).dropna()
        if len(diff) < 2:
            print(f"  {cond}: n too small")
            continue
        t, p = stats.ttest_1samp(diff.values, 0)
        sem = diff.std() / np.sqrt(len(diff))
        ci_lo = diff.mean() - 1.96 * sem
        ci_hi = diff.mean() + 1.96 * sem
        print(f"  {cond:8s}: Δ_round4_EV = {diff.mean():+.4f}  "
              f"95%CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  "
              f"t={t:+.2f}  p={p:.4f}  n={len(diff)}")

    # Sign test on (pos_all - neg_all)
    if "pos_all" in pivot.columns and "neg_all" in pivot.columns:
        contrast = (pivot["pos_all"] - pivot["neg_all"]).dropna()
        if len(contrast) >= 2:
            t, p = stats.ttest_1samp(contrast.values, 0)
            sem = contrast.std() / np.sqrt(len(contrast))
            ci_lo = contrast.mean() - 1.96 * sem
            ci_hi = contrast.mean() + 1.96 * sem
            n_pos = (contrast > 0).sum()
            print(f"\n  pos_all − neg_all on round_4 EV: "
                  f"Δ = {contrast.mean():+.4f}  "
                  f"95%CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  "
                  f"t={t:+.2f}  p={p:.4f}")
            print(f"  pos_all > neg_all in {n_pos}/{len(contrast)} agent-rows "
                  f"({100*n_pos/len(contrast):.0f}%)")


# Analysis 1: all agent-rows (baseline)
paired_test("ALL agent-rows (no filter)", df)

# Analysis 2: drop rows where BOTH conditions hit saturation
print("\n" + "=" * 64)
print("Saturation-filtered analyses")
print("=" * 64)

# For each scenario+agent_id, check max_p in each condition
maxp_pivot = df.pivot_table(
    index=["scenario_id", "agent_id"],
    columns="condition",
    values="round_4_maxp",
)

# Filter: keep agent-pairs where at least one condition is non-saturated
keep_any = (maxp_pivot < SAT_THRESHOLD).any(axis=1)
keep_both = (maxp_pivot < SAT_THRESHOLD).all(axis=1)

# Reattach to df
df = df.set_index(["scenario_id", "agent_id"])

df_any = df[df.index.isin(maxp_pivot[keep_any].index)].reset_index()
paired_test(f"Non-saturated in ANY condition (max_p<{SAT_THRESHOLD})", df_any)

df_both = df[df.index.isin(maxp_pivot[keep_both].index)].reset_index()
paired_test(f"Non-saturated in ALL conditions (max_p<{SAT_THRESHOLD})", df_both)

# Compare effect sizes
print("\n" + "=" * 64)
print("Effect size comparison (mean Δ round_4_EV vs control)")
print("=" * 64)
for label, dfx in [("all rows", df.reset_index()),
                    ("any non-sat", df_any),
                    ("both non-sat", df_both)]:
    pivot = dfx.pivot_table(index=["scenario_id", "agent_id"],
                             columns="condition",
                             values="round_4_ev")
    for cond in ["pos_all", "neg_all", "pos_r2"]:
        if cond not in pivot.columns:
            continue
        diff = (pivot[cond] - pivot["control"]).dropna()
        print(f"  {label:>15s} | {cond:>8s}: Δ = {diff.mean():+.4f} (n={len(diff)})")
    print()