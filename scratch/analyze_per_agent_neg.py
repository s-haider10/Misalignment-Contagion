"""Per-agent-within-scenario suppression analysis (neg_all only, no filter).

For each (scenario_id, agent_id) pair, compute round_4 EV under control
and under neg_all. The pair is the unit. Question: in how many of the
240 pairs did neg_all produce lower round_4 EV than control?

No saturation filter. No aggregation across agents. Every paired
observation counts.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

JSONL = Path("outputs/steering_experiment/trials.jsonl")


def ev(p):
    if not p:
        return float("nan")
    return sum(int(k) * float(v) for k, v in p.items())


# Load all agent-rows
rows = []
with open(JSONL) as f:
    for line in f:
        if not line.strip():
            continue
        tr = json.loads(line)
        for a in tr["agents"]:
            rows.append({
                "scenario_id": tr["scenario_id"],
                "condition": tr["condition"],
                "agent_id": a["agent_id"],
                "round_4_ev": ev(a["round_probs"][-1]),
            })

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} agent rows.")

# Pivot to (scenario_id, agent_id) × condition
pivot = df.pivot_table(
    index=["scenario_id", "agent_id"],
    columns="condition",
    values="round_4_ev",
)

# Drop pairs where either control or neg_all is missing
pairs = pivot[["control", "neg_all"]].dropna()
print(f"Paired observations (scenario × agent): {len(pairs)}\n")

# Per-pair difference
pairs = pairs.copy()
pairs["delta"] = pairs["neg_all"] - pairs["control"]

# Counts
n_total = len(pairs)
n_lower = (pairs["delta"] < 0).sum()
n_higher = (pairs["delta"] > 0).sum()
n_equal = (pairs["delta"] == 0).sum()

print("=" * 70)
print("Per-agent-within-scenario: did neg_all bring round_4 EV down?")
print("=" * 70)
print(f"\n  neg_all < control (suppression worked):     {n_lower:>4}/{n_total}  "
      f"({100*n_lower/n_total:.1f}%)")
print(f"  neg_all > control (steering wrong way):     {n_higher:>4}/{n_total}  "
      f"({100*n_higher/n_total:.1f}%)")
print(f"  neg_all = control (identical, often saturated): {n_equal:>4}/{n_total}  "
      f"({100*n_equal/n_total:.1f}%)")

# Sign test on non-tied pairs
n_directional = n_lower + n_higher
if n_directional > 0:
    binom_p = stats.binomtest(n_lower, n_directional, p=0.5, alternative="greater").pvalue
    print(f"\n  Sign test on the {n_directional} non-tied pairs:")
    print(f"  neg_all suppressed in {n_lower}/{n_directional} = "
          f"{100*n_lower/n_directional:.1f}%  (chance = 50%)")
    print(f"  Binomial p (one-sided, neg < ctrl): {binom_p:.4g}")

# Magnitude
print(f"\n  Mean Δ (neg_all - control) across all {n_total} pairs:  "
      f"{pairs['delta'].mean():+.4f} EV")
print(f"  Median Δ:                                            "
      f"{pairs['delta'].median():+.4f} EV")
print(f"  Std:                                                 "
      f"{pairs['delta'].std():.4f}")

# Among the pairs that DID move (Δ ≠ 0), how big was the move?
movers = pairs[pairs["delta"] != 0]
if len(movers) > 0:
    print(f"\n  Among the {len(movers)} pairs where neg_all and control differed:")
    print(f"    Mean |Δ|:                {movers['delta'].abs().mean():.4f} EV")
    print(f"    Mean Δ (signed):         {movers['delta'].mean():+.4f} EV")
    print(f"    Of these movers, how many went the right way (Δ<0)?  "
          f"{(movers['delta'] < 0).sum()}/{len(movers)} "
          f"({100*(movers['delta']<0).sum()/len(movers):.1f}%)")

# Per-scenario rollup: in each scenario, what fraction of agents showed suppression?
print("\n" + "=" * 70)
print("Per-scenario suppression rate (within-agent comparison)")
print("=" * 70)
print(f"\n{'scenario':<14} {'n_agents':>9} {'n_suppressed':>13} "
      f"{'%suppressed':>12} {'mean_Δ':>9} {'ctrl_EV':>9} {'neg_EV':>8}")

scen_summary = []
for scen_id in sorted(pairs.index.get_level_values(0).unique()):
    sub = pairs.xs(scen_id, level=0)
    n_agents = len(sub)
    n_supp = (sub["delta"] < 0).sum()
    n_eq = (sub["delta"] == 0).sum()
    mean_delta = sub["delta"].mean()
    mean_ctrl = sub["control"].mean()
    mean_neg = sub["neg_all"].mean()
    print(f"{scen_id:<14} {n_agents:>9} {n_supp:>13} "
          f"{100*n_supp/n_agents:>11.1f}%  "
          f"{mean_delta:>+9.3f} {mean_ctrl:>9.3f} {mean_neg:>8.3f}")
    scen_summary.append({
        "scenario_id": scen_id, "n_agents": n_agents, "n_suppressed": n_supp,
        "mean_delta": mean_delta, "ctrl_ev": mean_ctrl, "neg_ev": mean_neg,
    })

scen_df = pd.DataFrame(scen_summary)
n_scen_with_majority_suppression = (scen_df["n_suppressed"] > scen_df["n_agents"] / 2).sum()
n_scen_with_any_suppression = (scen_df["n_suppressed"] > 0).sum()
print(f"\n  Scenarios where >50% of agents got suppressed:    "
      f"{n_scen_with_majority_suppression}/{len(scen_df)}")
print(f"  Scenarios where ≥1 agent got suppressed:           "
      f"{n_scen_with_any_suppression}/{len(scen_df)}")
print(f"  Scenarios with mean Δ < 0 (suppression on average): "
      f"{(scen_df['mean_delta'] < 0).sum()}/{len(scen_df)}")

scen_df.to_csv("outputs/steering_experiment/per_scenario_neg_suppression.csv", index=False)
print(f"\nSaved → outputs/steering_experiment/per_scenario_neg_suppression.csv")